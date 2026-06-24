import re
from typing import List, Dict, Any

class LanguageProfile:
    def __init__(self, default_lang: str = "en-IN"):
        self.primary_language = default_lang
        self.secondary_language = None
        self.mix_ratio = 0  # percentage 0 to 100
        self.formality_level = "polite"  # casual / polite / formal
        self.history: List[Dict[str, Any]] = []
        
    def update(self, detected_language: str, confidence: float, transcript: str):
        """
        Update the rolling language profile with details from the latest customer utterance.
        """
        if not transcript:
            return
            
        # Check for explicit language preference mentions in customer transcript
        lower_transcript = transcript.lower()
        if "telugu" in lower_transcript:
            detected_language = "te-IN"
            confidence = 1.0
        elif "hindi" in lower_transcript:
            detected_language = "hi-IN"
            confidence = 1.0
        elif "english" in lower_transcript:
            detected_language = "en-IN"
            confidence = 1.0
            
        turn_data = {
            "detected_language": detected_language,
            "confidence": confidence,
            "transcript": transcript
        }
        self.history.append(turn_data)
        
        # 1. Update Primary and Secondary Languages based on recent history (last 5 turns)
        # We count frequencies but prioritize more recent languages in case of a tie.
        recent_turns = self.history[-5:]
        lang_stats = {}
        for idx, turn in enumerate(recent_turns):
            lang = turn["detected_language"]
            if lang not in lang_stats:
                lang_stats[lang] = {"count": 0, "last_index": idx}
            lang_stats[lang]["count"] += 1
            lang_stats[lang]["last_index"] = idx
            
        # Sort key: count first (primary), then last_index (secondary)
        sorted_langs = sorted(
            lang_stats.items(), 
            key=lambda x: (x[1]["count"], x[1]["last_index"]), 
            reverse=True
        )
        
        if sorted_langs:
            self.primary_language = sorted_langs[0][0]
            if len(sorted_langs) > 1:
                self.secondary_language = sorted_langs[1][0]
            else:
                self.secondary_language = None
                
        # 2. Heuristically calculate the mix ratio (percentage of English/ASCII words mixed in Indic speech)
        # If primary language is an Indic language (not en-IN), we check for English words in the text.
        # If transcript contains a mix of scripts (e.g. Telugu script + Latin script, or Devanagari + Latin),
        # or if it is written in Latin script but containing mixed English words.
        words = transcript.split()
        if words:
            # Check for words containing only letters/numbers/punctuation (ASCII) vs. native script
            # In India, ASRs like Sarvam can output transcripts in native script (like Telugu/Hindi script)
            # or in transliterated Latin script.
            # If native script: count ASCII words as English words.
            # If transliterated Latin: we can check for common English words.
            ascii_word_count = sum(1 for w in words if re.match(r'^[a-zA-Z0-9.,!?\'"\-]+$', w))
            
            # Simple heuristic:
            # If primary language is Indic (e.g. te-IN, hi-IN) and we have ASCII words, those are English words.
            if self.primary_language != "en-IN":
                self.mix_ratio = int((ascii_word_count / len(words)) * 100)
                # If we detect mixing and secondary language is not set, set it to English
                if self.mix_ratio > 10 and not self.secondary_language:
                    self.secondary_language = "en-IN"
            else:
                # If primary is English, check if there are non-ASCII words (representing Indic words)
                non_ascii_count = len(words) - ascii_word_count
                self.mix_ratio = int((non_ascii_count / len(words)) * 100)
                if self.mix_ratio > 10 and not self.secondary_language:
                    # Fallback default secondary language (e.g. Hindi or Telugu)
                    self.secondary_language = "te-IN"  # Defaulting to Telugu given Fuel Tracks region
                    
        # Clamp mix ratio
        self.mix_ratio = max(0, min(100, self.mix_ratio))
        
        # 3. Formality level detection (heuristic based on honorifics and keywords)
        # In Hindi/Telugu, honorifics like "andi", "garu", "aap", "ji" indicate polite/formal.
        # Casual terms like "tu", "re", "naa" indicate casual.
        lower_transcript = transcript.lower()
        
        formal_keywords = ["garu", "andi", "aap", "ji", "namaste", "namaskaram", "please", "thank you"]
        casual_keywords = ["tu", "tera", "re", "yaaro", "dude", "bro", "chafe", "raa", "rey"]
        
        formal_count = sum(1 for kw in formal_keywords if kw in lower_transcript)
        casual_count = sum(1 for kw in casual_keywords if kw in lower_transcript)
        
        if formal_count > casual_count:
            self.formality_level = "formal"
        elif casual_count > formal_count:
            self.formality_level = "casual"
        else:
            # Default to polite for customer calls
            self.formality_level = "polite"

    def get_summary(self) -> Dict[str, Any]:
        return {
            "primary_language": self.primary_language,
            "secondary_language": self.secondary_language,
            "mix_ratio": self.mix_ratio,
            "formality_level": self.formality_level
        }
