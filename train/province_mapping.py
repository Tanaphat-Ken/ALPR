"""
Complete mapping of Thai province codes to names and tokens.

This module provides comprehensive mappings for all 77 Thai provinces
including Bangkok and special administrative areas.
"""

# Complete mapping of Thai province codes to Thai names
PROVINCE_CODE_TO_THAI_NAME = {
    # Central Thailand
    "TH-10": "กรุงเทพมหานคร",  # Bangkok
    "TH-11": "สมุทรปราการ",    # Samut Prakan
    "TH-12": "นนทบุรี",       # Nonthaburi
    "TH-13": "ปทุมธานี",      # Pathum Thani
    "TH-14": "พระนครศรีอยุธยา", # Phra Nakhon Si Ayutthaya
    "TH-15": "อ่างทอง",       # Ang Thong
    "TH-16": "ลพบุรี",        # Lopburi
    "TH-17": "สิงห์บุรี",      # Sing Buri
    "TH-18": "ชัยนาท",       # Chai Nat
    "TH-19": "สระบุรี",       # Saraburi
    "TH-20": "ชลบุรี",        # Chonburi
    "TH-21": "ระยอง",         # Rayong
    "TH-22": "จันทบุรี",      # Chanthaburi
    "TH-23": "ตราด",         # Trat
    "TH-24": "ฉะเชิงเทรา",    # Chachoengsao
    "TH-25": "ปราจีนบุรี",     # Prachin Buri
    "TH-26": "นครนายก",      # Nakhon Nayok
    "TH-27": "สระแก้ว",       # Sa Kaeo
    
    # Northern Thailand
    "TH-50": "เชียงใหม่",      # Chiang Mai
    "TH-51": "ลำพูน",         # Lamphun
    "TH-52": "ลำปาง",        # Lampang
    "TH-53": "อุตรดิตถ์",     # Uttaradit
    "TH-54": "แพร่",         # Phrae
    "TH-55": "น่าน",         # Nan
    "TH-56": "พะเยา",        # Phayao
    "TH-57": "เชียงราย",      # Chiang Rai
    "TH-58": "แม่ฮ่องสอน",    # Mae Hong Son
    
    # Northeastern Thailand (Isan)
    "TH-30": "นครราชสีมา",    # Nakhon Ratchasima (Korat)
    "TH-31": "บุรีรัมย์",      # Buriram
    "TH-32": "สุรินทร์",      # Surin
    "TH-33": "ศรีสะเกษ",      # Sisaket
    "TH-34": "อุบลราชธานี",    # Ubon Ratchathani
    "TH-35": "ยโสธร",        # Yasothon
    "TH-36": "ชัยภูมิ",       # Chaiyaphum
    "TH-37": "อำนาจเจริญ",    # Amnat Charoen
    "TH-39": "หนองบัวลำภู",   # Nong Bua Lam Phu
    "TH-40": "ขอนแก่น",      # Khon Kaen
    "TH-41": "อุดรธานี",      # Udon Thani
    "TH-42": "เลย",          # Loei
    "TH-43": "หนองคาย",      # Nong Khai
    "TH-44": "มหาสารคาม",     # Maha Sarakham
    "TH-45": "ร้อยเอ็ด",      # Roi Et
    "TH-46": "กาฬสินธุ์",     # Kalasin
    "TH-47": "สกลนคร",       # Sakon Nakhon
    "TH-48": "นครพนม",       # Nakhon Phanom
    "TH-49": "มุกดาหาร",      # Mukdahan
    
    # Western Thailand
    "TH-70": "ราชบุรี",       # Ratchaburi
    "TH-71": "กาญจนบุรี",     # Kanchanaburi
    "TH-72": "สุพรรณบุรี",     # Suphan Buri
    "TH-73": "นครปฐม",       # Nakhon Pathom
    "TH-74": "สมุทรสาคร",     # Samut Sakhon
    "TH-75": "สมุทรสงคราม",   # Samut Songkhram
    "TH-76": "เพชรบุรี",      # Phetchaburi
    "TH-77": "ประจวบคีรีขันธ์", # Prachuap Khiri Khan
    
    # Southern Thailand
    "TH-80": "นครศรีธรรมราช",  # Nakhon Si Thammarat
    "TH-81": "กระบี่",        # Krabi
    "TH-82": "พังงา",        # Phang Nga
    "TH-83": "ภูเก็ต",        # Phuket
    "TH-84": "สุราษฎร์ธานี",   # Surat Thani
    "TH-85": "ระนอง",         # Ranong
    "TH-86": "ชุมพร",        # Chumphon
    "TH-90": "สงขลา",        # Songkhla
    "TH-91": "สตูล",         # Satun
    "TH-92": "ตรัง",         # Trang
    "TH-93": "พัทลุง",        # Phatthalung
    "TH-94": "ปัตตานี",       # Pattani
    "TH-95": "ยะลา",         # Yala
    "TH-96": "นราธิวาส",      # Narathiwat
    
    # Northern Central Thailand
    "TH-60": "นครสวรรค์",     # Nakhon Sawan
    "TH-61": "อุทัยธานี",      # Uthai Thani
    "TH-62": "กำแพงเพชร",     # Kamphaeng Phet
    "TH-63": "ตาก",          # Tak
    "TH-64": "สุโขทัย",       # Sukhothai
    "TH-65": "พิษณุโลก",      # Phitsanulok
    "TH-66": "พิจิตร",        # Phichit
    "TH-67": "เพชรบูรณ์",     # Phetchabun
}

def get_province_token(province_code: str, format_type: str = "code") -> str:
    """
    Get the appropriate province token for a given province code.
    
    Args:
        province_code: Thai province code (e.g., "TH-10")
        format_type: "code" for <TH-XX> format, "thai" for <thai_name> format
        
    Returns:
        Formatted province token
    """
    if format_type == "code":
        return f"<{province_code}>"
    elif format_type == "thai":
        thai_name = PROVINCE_CODE_TO_THAI_NAME.get(province_code, province_code)
        return f"<{thai_name}>"
    else:
        raise ValueError(f"Unknown format_type: {format_type}")

def get_all_province_tokens(format_type: str = "code") -> list[str]:
    """
    Get all possible province tokens for tokenizer expansion.
    
    Args:
        format_type: "code" for <TH-XX> format, "thai" for <thai_name> format
        
    Returns:
        List of all province tokens
    """
    if format_type == "code":
        return [f"<{code}>" for code in PROVINCE_CODE_TO_THAI_NAME.keys()]
    elif format_type == "thai":
        return [f"<{name}>" for name in PROVINCE_CODE_TO_THAI_NAME.values()]
    else:
        raise ValueError(f"Unknown format_type: {format_type}")

def get_special_tokens(format_type: str = "code") -> list[str]:
    """
    Get all special tokens needed for province prediction.
    
    Args:
        format_type: "code" for <TH-XX> format, "thai" for <thai_name> format
        
    Returns:
        List of special tokens including <prov> separator and all province tokens
    """
    tokens = ["<prov>"]
    tokens.extend(get_all_province_tokens(format_type))
    return tokens

# For backward compatibility
PROVINCE_NAMES = {code: f"<{name}>" for code, name in PROVINCE_CODE_TO_THAI_NAME.items()}