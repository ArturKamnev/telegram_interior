"""
Interior design Telegram bot using OpenRouter for both image generation
and approximate object detection. The bot leverages OpenRouter's
`google/gemini-2.0-flash-exp:free` model to generate photorealistic
interior renders and to analyse user-uploaded photos.

This script builds upon previous versions by introducing several major
enhancements:

* **Image generation via OpenRouter:** The bot uses OpenRouter's free
  `google/gemini-2.0-flash-exp:free` model to produce photorealistic interior renders.
  The `OPENROUTER_API_KEY` must grant access to this model. Generation requests
  are sent to the `/v1/images/generations` endpoint.
* **Customisable preferences:** The bot asks the user for primary and secondary colour
  preferences, optional additional elements to include in the scene, and the number of
  image variations to generate (1–4). These inputs are embedded into a detailed prompt
  that emphasises room dimensions to encourage the model to respect them.
* **Object detection using OpenRouter:** The `/detect` command attempts to identify
  objects in a user‑submitted photo via OpenRouter's chat completions endpoint. By
  default it uses the same `google/gemini-2.0-flash-exp:free` model; you can
  override the detection model via the `OPENROUTER_DETECTION_MODEL` environment
  variable. When detection succeeds, the bot returns the names of the objects along
  with pre‑built Google search links; otherwise it falls back to analysing the
  dominant colours and offers suggestions based on a predefined mapping of typical
  room features.

To run this bot you need the following packages:

```
pip install python-telegram-bot Pillow requests
```

Environment variables:

```
TELEGRAM_BOT_TOKEN   – your Telegram bot token from BotFather
OPENROUTER_API_KEY   – your OpenRouter API key for both image generation and image analysis
```

Limitations:

* Detection of objects in photos is approximate. The chosen OpenRouter model may produce
  generic or incomplete lists of objects, and the fallback colour‑based approach relies
  on a fixed mapping of room features.
* OpenRouter’s free tier may impose rate limits or quotas. Be sure your API key
  has sufficient access.
"""

import json
import logging
import os
from io import BytesIO
from typing import Dict, List, Tuple

import requests
from PIL import Image
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# ---------------------------------------------------------------------------
# Logging configuration
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("interior-bot-openrouter")


# ---------------------------------------------------------------------------
# Conversation state identifiers for the generation flow

STATE_AREA = 0
STATE_ROOM = 1
STATE_HEIGHT = 2
STATE_WIDTH = 3
STATE_COLOR_PRIMARY = 4
STATE_COLOR_SECONDARY = 5
STATE_ADDITIONS = 6
STATE_STYLE = 7
STATE_VARIATIONS = 8

# Conversation state identifiers for the detection flow
DETECT_WAIT_PHOTO = 100
DETECT_ROOM_TYPE = 101
DETECT_STYLE = 102


# ---------------------------------------------------------------------------
# Predefined data: room features and keyboards

ROOM_FEATURES: Dict[str, str] = {
    "Гостиная": "sofa, coffee table, TV zone, shelving, curtains",
    "Спальня": "double bed, nightstands, wardrobe, soft lighting",
    "Кухня": "modular cabinets, countertop, stove, sink, fridge; dining for 2–4",
    "Евро‑кухня (кухня‑гостиная)": "combined space: kitchen line/L‑shape, island or dining table, sofa and TV zone",
    "Столовая": "dining table, chairs, buffet/sideboard, pendant lighting",
    "Ванная": "vanity with sink, mirror, shower or bath, storage, moisture‑resistant finishes",
    "Санузел": "toilet, compact vanity, mirror, storage niche",
    "Кабинет": "desk, ergonomic chair, bookshelf, task lighting",
    "Детская": "bed, study desk, storage, playful accents, safe materials",
    "Прихожая/Холл": "shoe storage, closet, mirror, bench, hooks",
    "Гардеробная": "modular closet systems, hangers, drawers, mirror, lighting",
    "Лоджия/Балкон": "compact seating, plants, small table, insulated look",
    "Студия": "zoned plan: sleeping area, mini‑kitchen, compact dining, storage",
    "Игровая": "open area, soft flooring, storage for toys, lounge seating",
    "Мастерская": "workbench, tool wall, practical lighting, durable surfaces",
}

ROOM_KEYBOARD: List[List[str]] = [
    ["Гостиная", "Спальня", "Кухня"],
    ["Евро‑кухня (кухня‑гостиная)", "Столовая", "Кабинет"],
    ["Детская", "Ванная", "Санузел"],
    ["Прихожая/Холл", "Гардеробная", "Лоджия/Балкон"],
    ["Студия", "Игровая", "Мастерская"],
]

STYLE_KEYBOARD: List[List[str]] = [
    ["Минимализм", "Сканди", "Современный"],
    ["Лофт", "Классика", "Японский ваби‑саби"],
]


# ---------------------------------------------------------------------------
# Utility functions

def parse_float(value: str) -> float:
    """Attempt to parse a positive float from a string. Raises ValueError on failure."""
    try:
        val = float(value.replace(",", ".").strip())
        if val <= 0:
            raise ValueError
        return val
    except Exception as exc:
        raise ValueError("Invalid number") from exc


def extract_dominant_colors(image_bytes: bytes, k: int = 3) -> List[str]:
    """
    Extract the ``k`` most common colours from an image and return
    their hexadecimal codes (e.g. ``#aabbcc``).

    This helper remains in the codebase as a fallback if detection via
    Mistral fails.  It downsamples the image and counts the most
    frequent pixel colours.
    """
    with Image.open(BytesIO(image_bytes)) as img:
        img = img.convert("RGB").resize((100, 100))
        pixels = list(img.getdata())
    counts: Dict[Tuple[int, int, int], int] = {}
    for rgb in pixels:
        counts[rgb] = counts.get(rgb, 0) + 1
    top_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]
    return ["#%02x%02x%02x" % rgb for rgb, _ in top_colors]


def build_search_suggestions(room_type: str, style: str, colours: List[str]) -> List[Tuple[str, str]]:
    """
    Build search suggestions from room type, style and colours. Returns a list
    of (name, url) pairs.  The colour codes are used only in the query
    string; they are not included in the displayed name.
    """
    features_str = ROOM_FEATURES.get(room_type, "")
    raw_features = [f.strip() for part in features_str.split(";") for f in part.split(",") if f.strip()]
    suggestions: List[Tuple[str, str]] = []
    for feature in raw_features:
        if len(suggestions) >= 5:
            break
        for color in colours:
            if len(suggestions) >= 5:
                break
            query = f"{feature} {style} {color} купить интерьер"
            from urllib.parse import quote_plus
            url = f"https://www.google.com/search?q={quote_plus(query)}"
            # Only feature name (title‑cased) is displayed; colour is dropped from the name
            suggestions.append((feature.title(), url))
    return suggestions


def build_generation_prompt(data: Dict[str, str]) -> str:
    """
    Construct the prompt for image generation incorporating room
    information, colours and additions.  The prompt emphasises the
    dimensions and preferences to encourage the model to respect them.
    """
    area = data.get("area")
    room_type = data.get("room_type")
    height = data.get("height")
    width = data.get("width")
    primary_color = data.get("primary_color")
    secondary_color = data.get("secondary_color")
    additions = data.get("additions")
    style = data.get("style")
    features = ROOM_FEATURES.get(room_type, "functional layout appropriate to the room type")
    lines = []
    lines.append("Task: produce a photorealistic interior design render.")
    lines.append(f"Room type: {room_type}. Include: {features}.")
    lines.append(f"Style: {style}.")
    if primary_color:
        # Emphasise the primary colour as the dominant palette
        lines.append(f"Primary colour: {primary_color}. Use this hue as the dominant palette throughout the room.")
    if secondary_color and secondary_color.lower() not in {"", "нет", "none", "n/a"}:
        # Emphasise the secondary colour as an accent and instruct not to ignore it
        lines.append(f"Secondary colour: {secondary_color}. Use this as an accent colour complementing the primary; do not ignore it.")
    if additions and additions.lower() not in {"", "нет", "none", "n/a"}:
        # Mandate the inclusion of additional elements exactly once and warn against substitution
        lines.append(f"Mandatory additions: {additions}. Include each of these items exactly once; they must appear prominently and must not be replaced by other objects.")
    lines.append(f"Room data: area {area} m²; width {width} m; ceiling height {height} m.")
    lines.append(
        f"ABSOLUTE DIMENSIONS: area exactly {area} m²; width exactly {width} m; ceiling height exactly {height} m. "
        f"Do not alter these values; avoid double‑height ceilings or oversized windows; scale furniture realistically to fit."
    )
    lines.append(
        f"Repeat dimensions: {area} square meters area; width {width} m; height {height} m."
    )
    # Instruct the model to avoid duplicates of large items such as televisions or monitors
    lines.append("Avoid duplicates: do not include more than one television or screen; avoid adding objects not specified in the additions.")
    lines.append("One wide‑angle interior render (~22–24mm), camera at 1.5 m height; photorealistic lighting.")
    lines.append("Output: one final hero render of the entire room; no people, no watermarks.")
    return " ".join(lines)


def detect_objects_via_openrouter(image_bytes: bytes) -> List[str]:
    """
    Use OpenRouter to detect objects in an image.

    The image is encoded as a base64 data URI and sent to the chat
    completions endpoint with a prompt asking for a concise list of
    objects. The detection model is selected via the
    ``OPENROUTER_DETECTION_MODEL`` environment variable; by default it is
    ``google/gemini-2.0-flash-exp:free``. The function returns a list of
    detected object names. On any error or unexpected response format,
    an empty list is returned.
    """
    api_key = OPENROUTER_API_KEY
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is missing")
    import base64
    b64 = base64.b64encode(image_bytes).decode("ascii")
    data_uri = f"data:image/jpeg;base64,{b64}"
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    detection_model = os.getenv("OPENROUTER_DETECTION_MODEL", "google/gemini-2.5-pro-exp-03-25")
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "List the main objects visible in this image. Return a concise comma-separated list without extra commentary.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                },
            ],
        }
    ]
    payload = {"model": detection_model, "messages": messages}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=120)
    except Exception as exc:
        logger.error("OpenRouter request failed: %s", exc)
        return []
    if resp.status_code != 200:
        logger.error("OpenRouter detection error %s: %s", resp.status_code, resp.text)
        return []
    try:
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return []
        content = choices[0].get("message", {}).get("content", "")
        items = [s.strip() for s in content.split(",") if s.strip()]
        return items
    except Exception as exc:
        logger.error("Error parsing OpenRouter response: %s", exc)
        return []


def generate_images_together(prompt: str, n: int, size: str = "1024x1024") -> List[bytes]:
    """
    Generate images using Together AI's FLUX model.

    This function sends a request to the ``/v1/images/generations`` endpoint
    of Together AI using the free ``black‑forest‑labs/FLUX.1-schnell-Free`` model.
    It returns a list of images (as bytes) according to the requested
    number ``n``.  If any error occurs, a RuntimeError is raised.
    """
    api_key = TOGETHER_API_KEY
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY environment variable is missing")
    model_name = "black-forest-labs/FLUX.1-schnell-Free"
    endpoint = "https://api.together.xyz/v1/images/generations"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "n": n,
        "size": size,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Together API error {resp.status_code}: {resp.text}")
    data = resp.json()
    images_data = data.get("data", [])
    if not images_data:
        raise RuntimeError("Together API returned no image data")
    results: List[bytes] = []
    for item in images_data:
        url = item.get("url")
        if not url:
            continue
        img_resp = requests.get(url, timeout=120)
        if img_resp.status_code == 200:
            results.append(img_resp.content)
    if not results:
        raise RuntimeError("No images were downloaded from Together AI")
    return results


# ---------------------------------------------------------------------------
# Generation conversation handlers

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /start command: begin the parameter collection."""
    context.user_data.clear()
    await update.message.reply_text(
        "Привет! Я помогу вам создать дизайн интерьера с учётом ваших параметров.\n"
        "Введите площадь комнаты в м² (например, 20.5).\n\n"
        "Вы можете отменить процесс в любой момент, отправив /cancel, или перейти к анализу фото командой /detect."
    )
    return STATE_AREA


async def area_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store area and ask for room type."""
    try:
        val = parse_float(update.message.text)
    except ValueError:
        await update.message.reply_text("Введите положительное число для площади.")
        return STATE_AREA
    context.user_data["area"] = val
    await update.message.reply_text(
        "Выберите тип комнаты:",
        reply_markup=ReplyKeyboardMarkup(ROOM_KEYBOARD, resize_keyboard=True, one_time_keyboard=True),
    )
    return STATE_ROOM


async def room_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store room type and ask for height."""
    context.user_data["room_type"] = update.message.text.strip()
    await update.message.reply_text(
        "Введите высоту потолка в метрах (например, 2.7):",
        reply_markup=ReplyKeyboardRemove(),
    )
    return STATE_HEIGHT


async def height_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store height and ask for width."""
    try:
        val = parse_float(update.message.text)
    except ValueError:
        await update.message.reply_text("Введите положительное число для высоты.")
        return STATE_HEIGHT
    context.user_data["height"] = val
    await update.message.reply_text("Введите ширину комнаты в метрах (например, 3.6):")
    return STATE_WIDTH


async def width_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store width and ask for primary colour."""
    try:
        val = parse_float(update.message.text)
    except ValueError:
        await update.message.reply_text("Введите положительное число для ширины.")
        return STATE_WIDTH
    context.user_data["width"] = val
    await update.message.reply_text(
        "Укажите основной (primary) цвет, который вы хотите видеть в интерьере.\n"
        "Например: бежевый, белый, серый. Можете использовать HEX-код."
    )
    return STATE_COLOR_PRIMARY


async def color_primary_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store primary colour and ask for secondary colour."""
    context.user_data["primary_color"] = update.message.text.strip()
    await update.message.reply_text(
        "Укажите дополнительный (secondary) цвет. Если второстепенный цвет не нужен, введите 'нет'."
    )
    return STATE_COLOR_SECONDARY


async def color_secondary_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store secondary colour and ask for additions."""
    context.user_data["secondary_color"] = update.message.text.strip()
    await update.message.reply_text(
        "Перечислите дополнительные элементы или особенности, которые вы хотите обязательно видеть в изображении.\n"
        "Например: живые растения, камин, панорамные окна. Если ничего особенного не нужно, введите 'нет'."
    )
    return STATE_ADDITIONS


async def additions_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store additions and ask for style."""
    context.user_data["additions"] = update.message.text.strip()
    await update.message.reply_text(
        "Выберите стиль интерьера:",
        reply_markup=ReplyKeyboardMarkup(STYLE_KEYBOARD, resize_keyboard=True, one_time_keyboard=True),
    )
    return STATE_STYLE


async def style_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store style and ask for number of variations."""
    context.user_data["style"] = update.message.text.strip()
    await update.message.reply_text(
        "Сколько вариаций изображения сгенерировать? Введите число от 1 до 4."
    )
    return STATE_VARIATIONS


async def variations_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Final step: parse the number of variations, generate images via OpenRouter,
    and send them to the user.
    """
    text = update.message.text.strip()
    try:
        n = int(text)
        if n < 1 or n > 4:
            raise ValueError
    except Exception:
        await update.message.reply_text("Введите целое число от 1 до 4.")
        return STATE_VARIATIONS
    context.user_data["variations"] = n
    await update.message.reply_text("Генерирую изображение… пожалуйста, подождите.")
    # Build prompt and generate images
    prompt = build_generation_prompt(context.user_data)
    try:
        # Generate each variation sequentially to ensure multiple unique outputs
        for idx in range(1, n + 1):
            imgs = generate_images_together(prompt, 1)
            # imgs may contain one image; send each
            for img_bytes in imgs:
                await update.message.reply_photo(
                    photo=img_bytes,
                    caption=f"Вариация {idx} из {n}" if n > 1 else None,
                )
    except Exception as exc:
        logger.exception("Image generation error")
        await update.message.reply_text(f"Не удалось сгенерировать изображение: {exc}")
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /cancel: terminate any conversation."""
    await update.message.reply_text(
        "Операция отменена. Чтобы начать заново, отправьте /start.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


# ---------------------------------------------------------------------------
# Detection conversation handlers

async def detect_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Entry point for /detect: prompt for an image."""
    await update.message.reply_text(
        "Отправьте фотографию интерьера, для которого вы хотите получить названия элементов и ссылки.",
    )
    return DETECT_WAIT_PHOTO


async def detect_photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Receive photo and ask for room type."""
    photo = update.message.photo[-1]
    photo_file = await photo.get_file()
    image_byte_array = await photo_file.download_as_bytearray()
    context.user_data["detect_image"] = bytes(image_byte_array)
    await update.message.reply_text(
        "Какой это тип комнаты? Выберите или введите ваш вариант:",
        reply_markup=ReplyKeyboardMarkup(ROOM_KEYBOARD, resize_keyboard=True, one_time_keyboard=True),
    )
    return DETECT_ROOM_TYPE


async def detect_room_type_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store detection room type and ask for style."""
    context.user_data["detect_room_type"] = update.message.text.strip()
    await update.message.reply_text(
        "В каком стиле интерьер? Выберите стиль или введите свой:",
        reply_markup=ReplyKeyboardMarkup(STYLE_KEYBOARD, resize_keyboard=True, one_time_keyboard=True),
    )
    return DETECT_STYLE


async def detect_style_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Complete detection: extract colours, build suggestions and respond
    with names and links.
    """
    style = update.message.text.strip()
    image_bytes = context.user_data.get("detect_image")
    room_type = context.user_data.get("detect_room_type", "Гостиная")
    if not image_bytes:
        await update.message.reply_text("Изображение не найдено. Попробуйте заново.")
        return ConversationHandler.END
    # First try to detect objects via OpenRouter
    object_names: List[str] = []
    try:
        object_names = detect_objects_via_openrouter(image_bytes)
    except Exception as exc:
        logger.exception("OpenRouter detection failed")
        object_names = []
    if object_names:
        # Build search links for each detected object
        lines = ["Обнаруженные объекты и поисковые ссылки:"]
        from urllib.parse import quote_plus
        for item in object_names[:5]:
            query = f"{item} {style} купить интерьер"
            url = f"https://www.google.com/search?q={quote_plus(query)}"
            lines.append(f"• {item.title()} — {url}")
        await update.message.reply_text(
            "\n".join(lines),
            disable_web_page_preview=False,
            reply_markup=ReplyKeyboardRemove(),
        )
        return ConversationHandler.END
    # If OpenRouter detection yields no objects, fall back to colour-based suggestions
    try:
        colours = extract_dominant_colors(image_bytes, 3)
    except Exception as exc:
        logger.exception("Colour extraction failed")
        await update.message.reply_text(f"Не удалось обработать изображение: {exc}")
        return ConversationHandler.END
    suggestions = build_search_suggestions(room_type, style, colours)
    if not suggestions:
        await update.message.reply_text(
            "Не удалось сформировать предложения для этого типа комнаты. Попробуйте другой тип."
        )
        return ConversationHandler.END
    colour_list = ", ".join(colours)
    lines = [f"Основные цвета: {colour_list}."]
    lines.append("Предлагаемые элементы:")
    for name, url in suggestions:
        lines.append(f"• {name} — {url}")
    await update.message.reply_text(
        "\n".join(lines),
        disable_web_page_preview=False,
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


# ---------------------------------------------------------------------------
# Application entry point

def main() -> None:
    token = TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN must be set")
    app = ApplicationBuilder().token(token).build()
    # Generation conversation definition
    gen_conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            STATE_AREA: [MessageHandler(filters.TEXT & ~filters.COMMAND, area_handler)],
            STATE_ROOM: [MessageHandler(filters.TEXT & ~filters.COMMAND, room_handler)],
            STATE_HEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, height_handler)],
            STATE_WIDTH: [MessageHandler(filters.TEXT & ~filters.COMMAND, width_handler)],
            STATE_COLOR_PRIMARY: [MessageHandler(filters.TEXT & ~filters.COMMAND, color_primary_handler)],
            STATE_COLOR_SECONDARY: [MessageHandler(filters.TEXT & ~filters.COMMAND, color_secondary_handler)],
            STATE_ADDITIONS: [MessageHandler(filters.TEXT & ~filters.COMMAND, additions_handler)],
            STATE_STYLE: [MessageHandler(filters.TEXT & ~filters.COMMAND, style_handler)],
            STATE_VARIATIONS: [MessageHandler(filters.TEXT & ~filters.COMMAND, variations_handler)],
        },
        fallbacks=[CommandHandler("cancel", cancel), CommandHandler("detect", detect_command)],
    )
    app.add_handler(gen_conv)
    # Detection conversation definition
    detect_conv = ConversationHandler(
        entry_points=[CommandHandler("detect", detect_command)],
        states={
            DETECT_WAIT_PHOTO: [MessageHandler(filters.PHOTO, detect_photo_handler)],
            DETECT_ROOM_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, detect_room_type_handler)],
            DETECT_STYLE: [MessageHandler(filters.TEXT & ~filters.COMMAND, detect_style_handler)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    app.add_handler(detect_conv)
    # Add global cancel for both conversations
    app.add_handler(CommandHandler("cancel", cancel))
    logger.info("Interior design bot (OpenRouter) started")
    app.run_polling()


if __name__ == "__main__":
    main()
