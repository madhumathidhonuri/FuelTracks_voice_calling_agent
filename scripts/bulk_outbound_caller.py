import os
import sys
import argparse
import asyncio
import logging
import openpyxl
from typing import Dict, Any, List

# Adjust path to import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.telephony.outbound_caller import OutboundCaller

# Setup CLI Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("bulk_outbound_caller")

def parse_excel_file(file_path: str, default_call_type: str = "lead_followup", default_product: str = None) -> List[Dict[str, Any]]:
    """
    Parses an Excel file containing customer data.
    Identifies headers case-insensitively and maps columns to customer details.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found at: {file_path}")

    try:
        workbook = openpyxl.load_workbook(file_path, data_only=True)
        sheet = workbook.active
        logger.info(f"Loaded Excel sheet: '{sheet.title}' from {file_path}")
    except Exception as e:
        raise ValueError(f"Failed to open/parse Excel file: {e}")

    # Read rows
    rows = list(sheet.iter_rows(values_only=True))
    if not rows:
        raise ValueError("Excel file is empty.")

    # Find headers in the first row
    headers = [str(cell).strip().lower().replace("_", "").replace(" ", "") if cell is not None else "" for cell in rows[0]]
    
    # Map headers to indices
    name_idx = -1
    phone_idx = -1
    product_idx = -1
    type_idx = -1

    name_headers = ["name", "customername", "recipientname"]
    phone_headers = ["phone", "number", "phonenumber", "customernumber", "mobile", "mobilenumber"]
    product_headers = ["product", "interest", "productinterest"]
    type_headers = ["type", "calltype", "purpose"]

    for idx, header in enumerate(headers):
        if any(h in header for h in name_headers):
            name_idx = idx
        elif any(h in header for h in phone_headers):
            phone_idx = idx
        elif any(h in header for h in product_headers):
            product_idx = idx
        elif any(h in header for h in type_headers):
            type_idx = idx

    # Validate header mappings
    if phone_idx == -1:
        raise ValueError(
            "Could not identify the phone number column. "
            "Please ensure your sheet has a header named 'Phone', 'Number', 'Mobile', or 'Phone Number'."
        )
    if name_idx == -1:
        logger.warning("Could not identify the customer name column. Defaulting customer names to 'Customer'.")

    customers = []
    # Read data rows (skipping header)
    for row_num, row in enumerate(rows[1:], start=2):
        # Skip empty rows
        if all(cell is None for cell in row):
            continue

        phone = str(row[phone_idx]).strip() if row[phone_idx] is not None else ""
        if not phone:
            logger.warning(f"Row {row_num}: Skipping due to empty phone number.")
            continue

        name = str(row[name_idx]).strip() if (name_idx != -1 and row[name_idx] is not None) else "Customer"
        product = str(row[product_idx]).strip() if (product_idx != -1 and row[product_idx] is not None) else default_product
        call_type = str(row[type_idx]).strip() if (type_idx != -1 and row[type_idx] is not None) else default_call_type

        # Validate call type
        if call_type not in ["lead_followup", "support", "dealer_recruitment", "marketing"]:
            logger.warning(f"Row {row_num}: Invalid call type '{call_type}'. Defaulting to '{default_call_type}'.")
            call_type = default_call_type

        customers.append({
            "name": name,
            "phone": phone,
            "product_interest": product,
            "call_type": call_type,
            "row_num": row_num
        })

    logger.info(f"Successfully parsed {len(customers)} customers from Excel.")
    return customers

async def main_async():
    parser = argparse.ArgumentParser(description="Bulk Outbound Call Initiator from Excel File")
    parser.add_argument(
        "--file", "-f", 
        required=True, 
        help="Path to the Excel file containing customer data (.xlsx)"
    )
    parser.add_argument(
        "--type", "-t",
        choices=["lead_followup", "support", "dealer_recruitment", "marketing"],
        help="Default call type to use for the campaign if not specified in the Excel sheet."
    )
    parser.add_argument(
        "--product", "-p",
        help="Default product interest if not specified in the Excel sheet."
    )
    args = parser.parse_args()

    # Infer default call type from file name if not explicitly provided
    default_call_type = args.type
    if not default_call_type:
        filename_lower = os.path.basename(args.file).lower()
        if "marketing" in filename_lower or "promo" in filename_lower:
            default_call_type = "marketing"
        elif "support" in filename_lower or "service" in filename_lower:
            default_call_type = "support"
        elif "dealer" in filename_lower or "recruit" in filename_lower or "partner" in filename_lower:
            default_call_type = "dealer_recruitment"
        else:
            default_call_type = "lead_followup"

    try:
        customers = parse_excel_file(args.file, default_call_type=default_call_type, default_product=args.product)
    except Exception as e:
        logger.error(f"Error parsing Excel: {e}")
        sys.exit(1)

    if not customers:
        logger.info("No customers to call. Exiting.")
        sys.exit(0)

    caller = OutboundCaller()
    success_count = 0
    failed_count = 0

    print("\n" + "="*60)
    print(f"Starting Outbound Campaign for {len(customers)} Customers")
    print("="*60)

    for cust in customers:
        name = cust["name"]
        phone = cust["phone"]
        prod = cust["product_interest"]
        c_type = cust["call_type"]
        row = cust["row_num"]

        print(f"\n[Row {row}] Calling {name} at {phone} (Type: {c_type}, Product: {prod or 'None'})...")
        
        try:
            result = await caller.initiate_call(
                customer_number=phone,
                customer_name=name,
                product_interest=prod,
                call_type=c_type
            )
            
            if result.get("success"):
                print(f"  [SUCCESS] Call SID: {result['call_sid']} (Status: {result['status']})")
                success_count += 1
            else:
                print(f"  [FAILED] {result.get('error')} (Status Code: {result.get('status_code', 'N/A')})")
                failed_count += 1
                
        except Exception as e:
            print(f"  [ERROR] Unexpected exception: {e}")
            failed_count += 1

        # Small delay between triggering calls to avoid spamming the gateway
        await asyncio.sleep(1.0)

    print("\n" + "="*60)
    print("Campaign Complete!")
    print(f"Successful Calls Triggered: {success_count}")
    print(f"Failed Calls Triggered:      {failed_count}")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nCampaign interrupted by user.")
        sys.exit(1)
