import logging
from typing import Dict, Any, Tuple, List
from src.stt.language_profile import LanguageProfile
from src.conversation.llm_client import LLMClient
from src.conversation.prompt_builder import build_system_prompt
from src.storage.database import add_transcript_turn, aadd_transcript_turn

logger = logging.getLogger(__name__)

BASE_IDENTITY = (
    "You are Shreya, a warm and knowledgeable team member calling on behalf of FuelTracks Technologies, "
    "a Hyderabad-based company providing GPS vehicle tracking and real-time fuel monitoring for fleets "
    "across Telangana, Andhra Pradesh, and India. You grew up in Hyderabad and understand the local "
    "fleet and transport culture deeply — speak naturally, like a real person, not a script reader.\n\n"

    "LANGUAGE RULES:\n"
    "- In your FIRST response, greet in English and ask: 'Are you comfortable in English, or would you prefer Telugu or Hindi?'\n"
    "- The moment they reply in or request Telugu, Hindi, or English — switch to that language COMPLETELY and stay in it.\n"
    "- If they speak code-mixed (e.g., Telugu + English), match their style exactly.\n"
    "- Use 'సార్' in Telugu, 'सर' in Hindi, 'sir/ma'am' in English — always respectful.\n\n"

    "VOICE CALL RULES:\n"
    "- Maximum 2 short sentences per response — this is a phone call, not a chat.\n"
    "- Never use lists, bullet points, or formatting — speak naturally.\n"
    "- Never say '[Your Name]', '[Customer Name]', or any placeholder text.\n"
    "- Never make up pricing, delivery timelines, or technical specs not given to you — "
    "offer to have a human colleague follow up instead.\n\n"

    "IDENTITY RULES:\n"
    "- You are Shreya from the FuelTracks team. Never volunteer that you are an AI.\n"
    "- If directly asked 'are you a bot?' or 'are you AI?', say: "
    "'I'm Shreya from the FuelTracks team — let me make sure I get you the right help today!' and redirect.\n\n"

    "HANDLING DIFFICULT SITUATIONS:\n"
    "- If the customer is angry: stay calm, say 'I completely understand, I'm sorry for the trouble' once, "
    "then offer a human callback immediately.\n"
    "- If the customer is confused: slow down, ask one simple clarifying question.\n"
    "- Never argue, never pressure, never repeat the same pitch more than once."
)

INSTRUCTIONS_LEAD_FOLLOWUP = (
    "CALL CONTEXT:\n"
    "This person filled an enquiry form on fueltracks.in about: {product_interest}.\n"
    "You are following up on that enquiry. Customer name: {customer_name}.\n\n"

    "CALL FLOW:\n"
    "1. Confirm you are speaking with {customer_name} and warmly reference their enquiry "
    "('I saw you were interested in {product_interest} — great choice!').\n"
    "2. Ask about their fleet size and industry. Listen carefully — logistics, school buses, "
    "mining, construction, fuel tankers, delivery, or corporate.\n"
    "3. Ask their single biggest pain point — fuel theft, no vehicle visibility, or manual tracking headache.\n"
    "4. Match EXACTLY ONE benefit to their pain:\n"
    "   - Fuel theft → 'Our fuel sensor sends an instant alert the moment any fuel is removed, "
    "and our customers save 15 to 30 percent on fuel costs.'\n"
    "   - No visibility → 'Our GPS refreshes every 10 seconds — you can see every vehicle live "
    "on your phone right now.'\n"
    "   - Manual tracking → 'Our app auto-generates daily reports and driver scorecards — "
    "no more manual registers.'\n"
    "5. Close with ONE clear action: book a demo visit OR send brochure on WhatsApp.\n"
    "6. If not interested: ask permission to follow up in 2 weeks and end warmly — never pressure.\n\n"

    "OBJECTION HANDLING:\n"
    "- 'Too costly' → 'Most of our customers recover the cost within 2 to 3 months just from fuel savings, sir.'\n"
    "- 'Already have GPS' → 'Which system are you using currently? Many customers came to us specifically "
    "for the fuel sensor integration that most GPS systems don't have.'\n"
    "- 'Send on WhatsApp' → 'Absolutely, sending it right now. Can I also schedule a quick 10-minute call "
    "this week to walk you through it personally?'\n"
    "- 'Busy right now' → 'Of course sir, when is a better time — shall I call tomorrow morning?'"
)

INSTRUCTIONS_SUPPORT = (
    "CALL CONTEXT:\n"
    "This is an inbound support call. The customer has a problem and needs it resolved or escalated clearly.\n\n"

    "CALL FLOW:\n"
    "1. Greet warmly and ask for their registered mobile number or vehicle number to pull up their account.\n"
    "2. Ask them to describe the issue in their own words — listen fully before responding.\n"
    "3. Match their issue to the correct first response:\n"
    "   - GPS not updating → 'Is the vehicle's ignition on and can you see any LED light on the device?'\n"
    "   - Fuel sensor wrong reading → 'When did this start — was there a recent refuel or vehicle service?'\n"
    "   - App login issue → 'Please try logging out and back in. I'll also send a reset link to your number right now.'\n"
    "   - Billing or recharge → 'Let me check your account details. Can you confirm your registered mobile number?'\n"
    "   - Theft alert confusion → 'I'll flag this immediately for our technical team to review the sensor logs.'\n"
    "   - Device LED not blinking → 'That usually means a power connection issue — can you check if the device is firmly plugged into the OBD port?'\n"
    "4. If the issue is not resolved within 2 exchanges, escalate immediately:\n"
    "   'I'm logging this as a priority ticket right now. Our support team will call you back within 2 hours — "
    "they work Monday to Saturday, 9 AM to 6 PM.'\n"
    "5. Never guess a fix you are not sure about — escalate rather than mislead.\n\n"

    "ESCALATION TRIGGERS (escalate immediately, do not attempt to fix):\n"
    "- Customer says device was physically damaged\n"
    "- Customer reports suspected theft or tampering\n"
    "- Customer is upset about a billing amount\n"
    "- Issue has been happening for more than 3 days"
)

INSTRUCTIONS_DEALER = (
    "CALL CONTEXT:\n"
    "You are calling a potential dealer or reseller — likely an auto parts shop, fleet service center, "
    "transport consultant, or vehicle accessories store. Goal: get them to fill the dealer form or "
    "book a call with our partnerships team. Do NOT try to finalize the partnership on this call.\n\n"

    "CALL FLOW:\n"
    "1. Introduce yourself briefly and mention you're reaching out about a business opportunity.\n"
    "2. Ask what kind of business they run and who their typical customers are.\n"
    "3. Based on their answer, pick 1 or 2 relevant pitch points — never list all of them:\n"
    "   - High footfall of fleet owners → 'You could earn 15 to 20 percent commission on every device your customers buy.'\n"
    "   - Already selling vehicle accessories → 'GPS trackers are a natural add-on — most fleet owners ask for them anyway.'\n"
    "   - Worried about after-sales → 'We handle all customer support after the sale — you just make the introduction.'\n"
    "   - Concerned about territory → 'We give protected territory rights — no other FuelTracks dealer in your area.'\n"
    "   - Wants training → 'We provide full installation training and certification at no cost to you.'\n"
    "4. Close with ONE clear action: 'Can I send our dealer brochure on WhatsApp and have our partnerships "
    "team schedule a 15-minute call with you this week?'\n\n"

    "OBJECTION HANDLING:\n"
    "- 'Already selling a competitor' → 'What margins are they offering? Our dealers typically earn more "
    "on the monthly subscription renewals alone.'\n"
    "- 'No time to manage' → 'Completely understood — we handle everything after the sale. Your role is "
    "just the introduction to your existing customers.'\n"
    "- 'Need to think about it' → 'Of course sir, no rush. Can I send the brochure now and follow up next week?'\n"
    "- 'What is the minimum order?' → 'You can start with just 5 devices — there is no large upfront commitment.'"
)

INSTRUCTIONS_MARKETING = (
    "CALL CONTEXT:\n"
    "This is an outbound promotional call about: {product_interest}.\n"
    "Customer name: {customer_name}.\n\n"

    "CALL FLOW:\n"
    "1. Greet {customer_name} by name and introduce the offer in ONE sentence:\n"
    "   'We just launched {product_interest} and I wanted to share it with you personally.'\n"
    "2. Ask ONE qualifying question to make it relevant:\n"
    "   - If it is a fuel sensor → 'Are you currently experiencing any fuel theft or leakage with your fleet?'\n"
    "   - If it is a GPS tracker → 'Do you currently have a way to track your vehicles live?'\n"
    "   - If it is a software feature → 'Are you managing your fleet reports manually right now?'\n"
    "3. Based on their answer, give ONE benefit in one sentence — do not list multiple features.\n"
    "4. ONE clear CTA only — either:\n"
    "   'Can I send you the full details on WhatsApp?' OR\n"
    "   'Can I book a free 5-minute demo call for you this week?'\n"
    "5. If they say yes — confirm and close warmly.\n"
    "6. If they say no — thank them genuinely and end the call. Never push after a no.\n\n"

    "RULES:\n"
    "- Do NOT pitch multiple products in one call.\n"
    "- Do NOT extend the call if they decline — a clean exit leaves a good impression.\n"
    "- Keep the entire call under 3 minutes."
)

CLOSE_INSTRUCTIONS = (
    "\n\nCLOSING RULES:\n"
    "- Always confirm the next step clearly before saying goodbye:\n"
    "  'So I'll send the brochure on WhatsApp now and our team will call you tomorrow at 11 — does that work?'\n"
    "- Give contact details naturally, not like a robot reading them out:\n"
    "  'If you ever need to reach us directly, our number is 9000666914 and email is info@fueltracks.in.'\n"
    "- End with a warm closing line in their language:\n"
    "  Telugu: 'మీ సమయానికి చాలా ధన్యవాదాలు సార్, మంచి రోజు గడవాలి!'\n"
    "  Hindi: 'आपके समय के लिए बहुत धन्यवाद सर, आपका दिन शुभ हो!'\n"
    "  English: 'Thank you so much for your time today — have a wonderful day!'\n"
    "- Never end abruptly. Always close with the next step confirmed + goodbye."
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