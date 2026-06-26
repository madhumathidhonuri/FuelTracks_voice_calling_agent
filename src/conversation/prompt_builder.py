"""
Prompt Builder
---------------
Builds the final system prompt from language profile, company instructions,
and optional style-template files. Returns a SystemPrompt dataclass so the
Groq LLM client has one consistent object — no risk of divergence.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from config.settings import settings
from src.stt.language_profile import LanguageProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared dataclass consumed by both LLM clients
# ---------------------------------------------------------------------------

@dataclass
class SystemPrompt:
    """
    Thin wrapper around the final system prompt string.

    Both LLM clients read `.text` so they always use the same string,
    eliminating any risk of per-client prompt divergence.
    """
    text: str

    def __str__(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------

LANG_MAP = {
    "te-IN": "te.txt",
    "te": "te.txt",
    "hi-IN": "hi.txt",
    "hi": "hi.txt",
    "en-IN": "en.txt",
    "en": "en.txt",
}

LANG_NAMES = {
    "te-IN": "Telugu",
    "te": "Telugu",
    "hi-IN": "Hindi",
    "hi": "Hindi",
    "en-IN": "English",
    "en": "English",
}


def load_prompt_file(path: Path) -> str:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to read prompt file at {path}: {e}")
    return ""


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    company_name: str,
    call_purpose: str,
    language_profile: LanguageProfile,
    company_specific_instructions: str,
) -> SystemPrompt:
    """
    Build a dynamic system prompt based on settings, language profile, and
    company instructions.

    Returns:
        SystemPrompt — consumed by the Groq LLM client via `.text`.
    """
    base_prompt_path = settings.PROMPTS_DIR / "base_system_prompt.txt"
    base_template = load_prompt_file(base_prompt_path)

    if not base_template:
        # Hardcoded fallback if file is somehow missing
        base_template = (
            "You are a voice calling agent for {company_name}, handling {call_purpose}.\n\n"
            "The customer is currently speaking primarily in {primary_language}.\n"
            "{secondary_language_section}\n\n"
            "Respond in natural, spoken {primary_language} — the way people actually talk on "
            "phone calls in daily life, NOT formal, literary, or textbook {primary_language}.\n"
            "{code_mixing_section}\n\n"
            "Match the customer's formality level: {formality_level} (casual / polite / formal).\n"
            "Keep responses short and conversational — this is a phone call, not a written message.\n\n"
            "{company_specific_instructions}"
        )

    # 1. Resolve language naming for prompts
    primary_lang_name = LANG_NAMES.get(language_profile.primary_language, "English")

    # 2. Build conditional sections
    secondary_language_section = ""
    code_mixing_section = ""

    if language_profile.secondary_language:
        sec_lang_name = LANG_NAMES.get(language_profile.secondary_language, "English")
        secondary_language_section = (
            f"They naturally mix in {sec_lang_name} words/phrases "
            f"(estimated {language_profile.mix_ratio}% of speech)."
        )
        code_mixing_section = (
            f"Mix in common {sec_lang_name} words naturally, the same way the customer does.\n"
            f"Do not over-correct to 'pure' {primary_lang_name} — that sounds robotic and unnatural to the customer."
        )

    # 3. Load style template instructions if available
    style_file = LANG_MAP.get(language_profile.primary_language, "en.txt")
    style_path = settings.STYLE_TEMPLATES_DIR / style_file
    style_notes = load_prompt_file(style_path)

    full_instructions = company_specific_instructions
    if style_notes:
        full_instructions += f"\n\n[STYLE GUIDELINE]\n{style_notes}"

    # 4. Code-mixing language directive (Tenglish / Hinglish style)
    if primary_lang_name != "English":
        script_map = {"Telugu": "Telugu script (లిపి)", "Hindi": "Devanagari script (देवनागरी)"}
        script_note = script_map.get(primary_lang_name, f"{primary_lang_name} script")
        strict_directive = (
            f"\n\n[LANGUAGE STYLE RULE]\n"
            f"Speak in natural, everyday {primary_lang_name} the way people actually talk on phone calls "
            f"in India — NOT pure literary or textbook {primary_lang_name}.\n"
            f"- Mix in common English words naturally: product names, technical terms (GPS, tracker, "
            f"demo, software, delivery, account, WhatsApp, rupees, percent, etc.) should stay in English.\n"
            f"- Use {script_note} for {primary_lang_name} words, but do NOT force English sentences "
            f"into {primary_lang_name} script.\n"
            f"- Do NOT reply entirely in English unless the customer switches to English first.\n"
            f"- Do NOT open with English phrases like 'Okay, Telugu it is!' — just speak naturally.\n"
            f"- Aim for roughly 60-70% {primary_lang_name} words, 30-40% English words woven in naturally.\n"
            f"- Example good style (Telugu): 'మీ vehicle కి GPS tracker install చేసుకుంటే real-time tracking "
            f"చేయవచ్చు, demo చూడాలంటే చెప్పండి.'\n"
            f"- Example bad style (too pure): 'మీ వాహనమునకు స్థాన నిర్ధారణ పరికరమును అమర్చుకొనుము.'"
        )
        full_instructions += strict_directive


    # 5. Interpolate template
    try:
        prompt_text = base_template.format(
            company_name=company_name,
            call_purpose=call_purpose,
            primary_language=primary_lang_name,
            secondary_language_section=secondary_language_section,
            code_mixing_section=code_mixing_section,
            formality_level=language_profile.formality_level,
            company_specific_instructions=full_instructions,
        )
        return SystemPrompt(text=prompt_text)
    except Exception as e:
        logger.error(f"Error interpolating system prompt template: {e}")
        fallback = f"You are a helpful voice assistant for {company_name}. Speak in {primary_lang_name}."
        return SystemPrompt(text=fallback)
