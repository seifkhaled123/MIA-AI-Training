class ticketCodec:
    def __init__(self):
        # We use a hyphen to cleanly separate the ticket ID from the checksum
        self.separator = "-"

    def _generate_checksum(self, data):
        """
        CHECKSUM ALGORITHM EXPLANATION:
        This algorithm starts with the standard djb2 seed (5381). For every character 
        in the ticket ID, it takes the running checksum and applies a bitwise left 
        shift (<< 5) to rapidly scramble the binary data. It then applies a bitwise 
        XOR (^) against the checksum multiplied by a prime number (31) to cause 
        unpredictable value scattering, and finally adds the ASCII value of the character.
        
        A bitwise AND mask (& 0xFFFFFFFF) is applied at the end of the loop to simulate 
        a 32-bit integer limit, preventing Python from generating infinitely large 
        numbers. The final integer is converted into a Hexadecimal string.
        """
        checksum = 5381
        for char in data:
            # Shift left by 5, XOR with prime multiplier (31), add ASCII value
            checksum = (((checksum << 5) ^ (checksum * 31)) + ord(char)) & 0xFFFFFFFF
        
        # Convert to a hexadecimal string, remove the '0x' prefix, and make uppercase
        return hex(checksum)[2:].upper()

    def encode(self, ticket_id):
        # Generate the checksum and attach it to the ticket ID
        checksum = self._generate_checksum(ticket_id)
        return f"{ticket_id}{self.separator}{checksum}"

    def decode(self, barcode):
        # If the barcode doesn't even have our separator, it's instantly invalid
        if self.separator not in barcode:
            return "CORRUPTED TICKET"
            
        # Split the barcode into the ID portion and the attached checksum
        # Using rsplit(..., 1) ensures we only split on the LAST hyphen, 
        # in case the ticket ID itself naturally contained hyphens.
        ticket_id, attached_checksum = barcode.rsplit(self.separator, 1)
        
        # Recompute the checksum from scratch using the ticket ID portion
        recomputed_checksum = self._generate_checksum(ticket_id)
        
        # Compare the embedded checksum with our newly computed one
        if recomputed_checksum == attached_checksum:
            return ticket_id
        else:
            return "CORRUPTED TICKET"


# ==========================================
# TEST SCRIPT (To demonstrate requirements)
# ==========================================
if __name__ == "__main__":
    codec = ticketCodec()

    print("--- ENCODING SAMPLES ---")
    ticket1 = "MIA2026GATE7"
    ticket2 = "VIP-ARG-001"
    ticket3 = "STAND-B-ROW42"

    barcode1 = codec.encode(ticket1)
    barcode2 = codec.encode(ticket2)
    barcode3 = codec.encode(ticket3)

    print(f"Original: {ticket1:15} -> Barcode: {barcode1}")
    print(f"Original: {ticket2:15} -> Barcode: {barcode2}")
    print(f"Original: {ticket3:15} -> Barcode: {barcode3}")

    print("\n--- DECODING VALID BARCODES ---")
    print(f"Decoding '{barcode1}': {codec.decode(barcode1)}")
    print(f"Decoding '{barcode2}': {codec.decode(barcode2)}")

    print("\n--- DECODING CORRUPTED BARCODE ---")
    # Hand-corrupting one character (Changing GATE7 to GATE8)
    corrupted_barcode = barcode1.replace("GATE7", "GATE8")
    
    print(f"Attempting to decode corrupted barcode '{corrupted_barcode}'...")
    print(f"Result: {codec.decode(corrupted_barcode)}")