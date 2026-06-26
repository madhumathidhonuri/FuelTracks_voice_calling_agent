import logging
from typing import Dict, Any, Tuple, List
from src.stt.language_profile import LanguageProfile
from src.conversation.llm_client import LLMClient
from src.conversation.prompt_builder import build_system_prompt
from src.storage.database import add_transcript_turn, aadd_transcript_turn

logger = logging.getLogger(__name__)

BASE_IDENTITY = (
    "You are Shreya, calling on behalf of Fuel Tracks Technologies, a Hyderabad-based company "
    "providing GPS vehicle tracking and real-time fuel monitoring for fleets across "
    "Telangana, Andhra Pradesh, and India. You sound like a real, warm team member — "
    "not a script reader. Your name is Shreya; never output placeholders like '[Your Name]'. "
    "Keep sentences short and natural; this is a phone call, not an email. Never make up pricing, "
    "exact delivery timelines, or technical specs you weren't given — offer to have a human follow up instead.\n\n"
    "Crucial Call Flow Guidelines:\n"
    "1. The initial greeting has already greeted the customer and asked to confirm their identity (e.g., 'Am I speaking with [Name]?').\n"
    "2. If the customer has confirmed their identity (our current turn), your immediate next response MUST be: introduce yourself and ask for their language preference, exactly like: 'Hi [Name], my name is Shreya, and I'm calling from Fuel Tracks Technologies. Just before we go further, are you comfortable in English, or would you prefer Hindi or Telugu?'\n"
    "3. Once the customer specifies their language (English, Hindi, or Telugu), you MUST switch entirely to that language for all future responses. Deliver the promotion/pitch in their preferred language, keep it short and professional, and ask for their permission to send a brochure/details on WhatsApp.\n"
    "4. If they accept or decline, politely close the call and say goodbye (always end with a polite goodbye phrase such as 'Thank you for your time, goodbye' or 'have a great day'). Never pressure."
)

INSTRUCTIONS_LEAD_FOLLOWUP = (
    "This person filled out an enquiry form on fueltracks.in about: {product_interest}\n"
    "(GPS Tracking Device / Fuel Sensor / AIS-140 Device / Fleet Management Software / School Bus Tracking).\n\n"
    "1. Greet warmly, confirm you're speaking with {customer_name}, reference their enquiry.\n"
    "2. Ask about their fleet — how many vehicles, what industry (logistics, school "
    "buses, mining, construction, rentals, fuel tankers, delivery, corporate).\n"
    "3. Connect their need to a real benefit: fuel theft detection, sub-10-second GPS "
    "refresh, fleet analytics — pick what's relevant, don't recite all of them.\n"
    "4. Goal: book a demo or site visit, or confirm details over WhatsApp.\n"
    "5. If not interested now, ask permission to follow up later and end politely — never pressure."
)

INSTRUCTIONS_SUPPORT = (
    "1. Confirm which vehicle/account they're calling about.\n"
    "2. Common issues: GPS not updating, fuel sensor reading wrong, app login trouble, "
    "billing/recharge, unclear theft alert.\n"
    "3. Offer one or two basic checks first (e.g., \"Is the vehicle's ignition on and "
    "the device powered? Can you try closing and reopening the app?\").\n"
    "4. If unresolved, tell them you're logging it for the support team and they'll "
    "hear back — mention support runs Mon–Sat, 9AM–6PM.\n"
    "5. For anything urgent or you're unsure about, offer the WhatsApp/phone number rather than guessing."
)

INSTRUCTIONS_DEALER = (
    "1. Pitch briefly: high commissions on devices and recurring SaaS margin, "
    "installation training & certification, protected territory rights, marketing support.\n"
    "2. Ask about their current business and region to gauge fit.\n"
    "3. Goal: get them to apply via the dealer form or book a call with sales — don't "
    "try to close the partnership on this call."
)

INSTRUCTIONS_MARKETING = (
    "This call is a promotional introduction of our latest product: {product_interest}.\n\n"
    "1. Explain that we just launched {product_interest} and highlight the main benefit (e.g., it prevents fuel theft issues and saves up to 15% on vehicle fuel costs).\n"
    "2. Ask if they are currently tracking their fleet or if they face fuel theft issues.\n"
    "3. Goal: Ask for their permission to send a product brochure via WhatsApp or book a 5-minute video demo.\n"
    "4. Once they respond, thank them politely for their time, say goodbye, and end the conversation."
)

CLOSE_INSTRUCTIONS = (
    "\n\nAlways close with: company contact — +91 9000666914, info@fueltracks.in — and thank them for their time."
)

class ConversationManager:
    def __init__(self, call_sid: str):
        self.call_sid = call_sid
        self.call_type = None  # lead_followup, support, dealer_recruitment, or inbound_routing
        self.context: Dict[str, Any] = {}
        self.history: List[Dict[str, str]] = []
        self.language_profile = LanguageProfile()
        self.llm_client = LLMClient()
        self.company_name = "Fuel Tracks Technologies"
        
    def initialize_call(self, call_type: str, context: Dict[str, Any] = None):
        """
        Set up the conversation state.
        :param call_type: 'lead_followup', 'support', 'dealer_recruitment', 'marketing', or 'inbound_routing'
        """
        self.call_type = call_type
        self.context = context or {}
        self.history = []
        self.language_profile = LanguageProfile()
        
        # If it is inbound and we need to route, prepare the first turn
        if self.call_type == "inbound_routing":
            # Add system instruction explaining routing phase
            routing_guideline = (
                "You are an automated router for Fuel Tracks. Ask the customer a single, quick routing question "
                "such as: 'Are you calling about an existing account, or interested in GPS tracking for your fleet?' "
                "Keep it warm and concise."
            )
            self.history.append({"role": "system", "content": routing_guideline})
            
    def get_company_specific_instructions(self) -> str:
        instructions = BASE_IDENTITY + "\n\n"
        
        if self.call_type == "lead_followup":
            prod = self.context.get("product_interest", "GPS Tracking Device")
            cust = self.context.get("customer_name", "Valued Customer")
            instructions += INSTRUCTIONS_LEAD_FOLLOWUP.format(product_interest=prod, customer_name=cust)
        elif self.call_type == "support":
            instructions += INSTRUCTIONS_SUPPORT
        elif self.call_type == "dealer_recruitment":
            instructions += INSTRUCTIONS_DEALER
        elif self.call_type == "marketing":
            prod = self.context.get("product_interest", "New GPS Tracker Pro")
            cust = self.context.get("customer_name", "Valued Customer")
            instructions += INSTRUCTIONS_MARKETING.format(product_interest=prod, customer_name=cust)
        else:
            # Fallback/routing instructions
            instructions += (
                "Currently routing the customer call. Ask if they are calling about "
                "an existing account (support), fleet tracking (sales), or partner programs (dealer recruitment)."
            )
            
        instructions += CLOSE_INSTRUCTIONS
        return instructions

    async def add_customer_turn(self, text: str, detected_language: str = "en-IN", confidence: float = 1.0):
        """
        Register caller input, update profile, and resolve inbound routing if necessary.
        """
        # Save transcript to local DB
        await aadd_transcript_turn(self.call_sid, "customer", text, detected_language, confidence)
        
        # Update rolling language profile
        self.language_profile.update(detected_language, confidence, text)
        
        # Add to local conversation history
        self.history.append({"role": "customer", "content": text})
        
        # Resolve routing if in inbound_routing state
        if self.call_type == "inbound_routing":
            await self._resolve_inbound_routing(text)
            
    async def _resolve_inbound_routing(self, customer_input: str):
        """
        Classify customer input into support, lead_followup, or dealer_recruitment.
        """
        # 1. Simple fast keyword heuristics
        lower_input = customer_input.lower()
        support_keywords = ["support", "existing", "recharge", "alert", "login", "account", "sensor", "not working", "problem", "issue", "vehicle", "truck"]
        sales_keywords = ["gps", "tracking", "buy", "price", "cost", "quote", "fleet", "purchase", "interested", "demo", "software", "inquiry"]
        dealer_keywords = ["dealer", "partner", "distributor", "franchise", "margin", "commission", "business", "sell"]
        
        support_score = sum(1 for kw in support_keywords if kw in lower_input)
        sales_score = sum(1 for kw in sales_keywords if kw in lower_input)
        dealer_score = sum(1 for kw in dealer_keywords if kw in lower_input)
        
        if support_score > sales_score and support_score > dealer_score:
            self.call_type = "support"
            logger.info(f"Routed call {self.call_sid} to 'support' via keywords")
            return
        elif sales_score > support_score and sales_score > dealer_score:
            self.call_type = "lead_followup"
            logger.info(f"Routed call {self.call_sid} to 'lead_followup' via keywords")
            return
        elif dealer_score > support_score and dealer_score > sales_score:
            self.call_type = "dealer_recruitment"
            logger.info(f"Routed call {self.call_sid} to 'dealer_recruitment' via keywords")
            return
            
        # 2. LLM fallback classification
        system_prompt = (
            "You are a routing classifier for Fuel Tracks Technologies.\n"
            "Classify the customer's text into one of these labels: 'support', 'lead_followup', 'dealer_recruitment'.\n"
            "Return ONLY the exact label name in lower case, nothing else. No explanation, no punctuation.\n\n"
            "Guidelines:\n"
            "- 'support': Problems, issues, logins, recharges, existing vehicles, account questions.\n"
            "- 'lead_followup': Inquiries about buying, GPS trackers, school bus tracking, fleet management software, pricing, demos.\n"
            "- 'dealer_recruitment': Interest in becoming a dealer, partner, selling the products, commissions, partnerships."
        )
        
        test_messages = [{"role": "customer", "content": f"Classify this utterance: '{customer_input}'"}]
        try:
            label, _ = await self.llm_client.generate_response(system_prompt, test_messages)
            label = label.strip().lower()
            if label in ["support", "lead_followup", "dealer_recruitment"]:
                self.call_type = label
                logger.info(f"Routed call {self.call_sid} to '{label}' via LLM classification")
                return
        except Exception as e:
            logger.error(f"Routing classification failed: {e}")
            
        # Default fallback
        self.call_type = "support"
        logger.info(f"Routed call {self.call_sid} to default 'support'")

    async def generate_agent_response(self) -> Tuple[str, Dict[str, int]]:
        """
        Generate agent's response using prompt builder and LLM client.
        """
        # Determine purpose name
        purpose_map = {
            "lead_followup": "lead follow-up enquiry",
            "support": "customer support",
            "dealer_recruitment": "dealer partner recruitment",
            "marketing": "marketing campaign",
            "inbound_routing": "initial welcome routing"
        }
        call_purpose = purpose_map.get(self.call_type, "customer service")
        
        # Build prompt
        system_prompt = build_system_prompt(
            company_name=self.company_name,
            call_purpose=call_purpose,
            language_profile=self.language_profile,
            company_specific_instructions=self.get_company_specific_instructions()
        )
        
        # Generate response from LLM
        response_text, token_usage = await self.llm_client.generate_response(system_prompt, self.history)
        
        # Append agent response to history and DB
        self.history.append({"role": "agent", "content": response_text})
        await aadd_transcript_turn(self.call_sid, "agent", response_text)
        
        return response_text, token_usage
