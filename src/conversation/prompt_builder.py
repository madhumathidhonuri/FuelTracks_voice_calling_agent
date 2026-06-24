import logging
from pathlib import Path
from config.settings import settings
from src.stt.language_profile import LanguageProfile

logger = logging.getLogger(__name__)

# Map language codes to style template filenames
LANG_MAP = {
    "te-IN": "te.txt",
    "te": "te.txt",
    "hi-IN": "hi.txt",
    "hi": "hi.txt",
    "en-IN": "en.txt",
    "en": "en.txt"
}

def load_prompt_file(path: Path) -> str:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to read prompt file at {path}: {e}")
    return ""

def build_system_prompt(
    company_name: str,
    call_purpose: str,
    language_profile: LanguageProfile,
    company_specific_instructions: str
) -> str:
    """
    Build a dynamic system prompt based on settings, language profile, and company instructions.
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
    lang_names = {
        "te-IN": "Telugu",
        "te": "Telugu",
        "hi-IN": "Hindi",
        "hi": "Hindi",
        "en-IN": "English",
        "en": "English"
    }
    
    primary_lang_name = lang_names.get(language_profile.primary_language, "English")
    
    # 2. Build conditional sections
    secondary_language_section = ""
    code_mixing_section = ""
    
    if language_profile.secondary_language:
        sec_lang_name = lang_names.get(language_profile.secondary_language, "English")
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
    
    # If style notes exist, append them to the company instructions
    full_instructions = company_specific_instructions
    if style_notes:
        full_instructions += f"\n\n[STYLE GUIDELINE]\n{style_notes}"

    # 4. Interpolate template
    try:
        system_prompt = base_template.format(
            company_name=company_name,
            call_purpose=call_purpose,
            primary_language=primary_lang_name,
            secondary_language_section=secondary_language_section,
            code_mixing_section=code_mixing_section,
            formality_level=language_profile.formality_level,
            company_specific_instructions=full_instructions
        )
        return system_prompt
    except Exception as e:
        logger.error(f"Error interpolating system prompt template: {e}")
        # Return fallback system prompt on error
        return f"You are a helpful voice assistant for {company_name}. Speak in {primary_lang_name}."
