import re
from typing import List, Tuple, Optional, Union


months_to_int_map = {
    'Unknown': 0,
    'January': 1, 'Jan': 1, 'February': 2, 'Feb': 2, 'March': 3, 'Mar': 3,
    'April': 4, 'Apr': 4, 'May': 5, 'June': 6, 'Jun': 6, 'July': 7, 'Jul': 7,
    'August': 8, 'Aug': 8, 'September': 9, 'Sep': 9, 'October': 10, 'Oct': 10,
    'November': 11, 'Nov': 11, 'December': 12, 'Dec': 12,
    'Enero': 1, 'Ene': 1, 'Febrero': 2, 'Feb': 2, 'Marzo': 3, 'Mar': 3,
    'Abril': 4, 'Abr': 4, 'Mayo': 5, 'Junio': 6, 'Jun': 6, 'Julio': 7, 'Jul': 7,
    'Agosto': 8, 'Ago': 8, 'Septiembre': 9, 'Set': 9, 'Octubre': 10, 'Oct': 10,
    'Noviembre': 11, 'Nov': 11, 'Diciembre': 12, 'Dic': 12,
}


# Reverse mapping for month numbers to full English names only
int_to_month_map = {
    0: 'Unknown',
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December'
}


def parse_date_component(date_str: str, default_year: Optional[int] = None) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse date string to extract month and year.
    Returns (month_num, year) where month_num is 1-12 or None.
    """
    if not date_str or not isinstance(date_str, str):
        return None, None
    
    date_str = str(date_str).strip()
    if date_str.lower() in ['unknown', 'nan', 'none', '', 'n/a', 'null', ' ']:
        return None, None
    
    # Extract year (4-digit, 1000-2999)
    year_match = re.search(r'\b(1[0-9]{3}|2[0-9]{3})\b', date_str)
    year = int(year_match.group()) if year_match else None
    
    # Extract month - check all month names/abbreviations
    month = None
    for month_name, month_num in months_to_int_map.items():
        pattern = rf'\b{re.escape(month_name)}\b'
        if re.search(pattern, date_str, re.IGNORECASE):
            month = month_num
            break
    
    return month, year


def date_parser(date_str: str, reference_year: Optional[int] = None, default_month: str = 'Unknown') -> List[Union[str, int]]:
    '''
    Enhanced date parser for coffee quality dataset.
    Rules:
    - Month range (May-August) → [May, reference_year or 0]
    - Year range (2009/2010) → [default_month, 2009]
    - Single year (2014) → [default_month, 2014]
    - Month+year range (December 2009-March 2010) → [December, 2009]
    - Month only (May) → [May, reference_year or 0]
    '''
    month, year = parse_date_component(date_str)
    
    # If no month found, default to Unknown
    month_num = month if month is not None else 0
    
    # Determine year: use parsed year, then reference_year, then 0
    final_year = year if year is not None else (reference_year if reference_year is not None else 0)
    
    # Get month name from number (prefer full names)
    month_name = int_to_month_map.get(month_num, default_month)
    
    return [month_name, final_year]


def calculate_month_difference(start_info: List[Union[str, int]], end_info: List[Union[str, int]]) -> int:
    """
    Calculate the difference in months between two dates.
    Handles cases where year might be 0 (unknown).
    """
    start_month = months_to_int_map.get(start_info[0], 0)
    end_month = months_to_int_map.get(end_info[0], 0)
    
    start_year = start_info[1] or 0
    end_year = end_info[1] or 0
    
    # If either year is 0, we can't calculate meaningful year difference
    if start_year == 0 or end_year == 0:
        # If years are unknown, assume they're the same year
        year_diff_in_months = 0
    else:
        year_diff_in_months = (end_year - start_year) * 12
    
    difference = end_month - start_month + year_diff_in_months
    return max(difference, 0)  # Don't return negative differences

