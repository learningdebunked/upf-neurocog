"""
NOVA food classification for NHANES WWEIA food codes.
Maps 8-digit FNDDS/WWEIA food codes to NOVA groups 1-4.
Primary source: NOVA group assignments based on:
  - First 2 digits = WWEIA food category group
  - Keyword matching on food description
"""

import re

# WWEIA category group (first 2 digits of 8-digit food code) → NOVA group
# Based on USDA WWEIA 2011-2014 food category structure
WWEIA_NOVA_MAP = {
    # NOVA 1: Unprocessed/minimally processed foods
    11: 1,  # Milk and milk drinks
    12: 1,  # Cheese
    14: 1,  # Milk desserts, sauces, gravies
    # (some items in 14 are NOVA 3/4 — handled by keyword)
    21: 1,  # Eggs
    23: 1,  # Beef (whole cuts, ground)
    24: 1,  # Pork (whole cuts, ground)
    25: 1,  # Lamb, veal, game
    26: 1,  # Poultry
    27: 1,  # Seafood (whole/fillets)
    31: 1,  # Beans/peas, cooked (not canned in sauce)
    32: 1,  # Tofu, soy
    41: 1,  # Vegetables (plain, raw, cooked)
    42: 1,  # Vegetables mixed dishes — needs keyword
    51: 1,  # Fruits (fresh, frozen, canned in juice)
    52: 1,  # Fruit nectars — needs keyword
    61: 1,  # Grains, cooked (rice, oats, pasta plain)
    62: 2,  # Breads, rolls → NOVA 2/3 — keyword refined
    63: 3,  # Crackers/salty snacks → NOVA 3/4
    64: 3,  # Quick breads, muffins, biscuits
    65: 3,  # Cakes, cookies, pastries
    66: 3,  # Pies, cobblers
    67: 3,  # Pancakes, waffles
    68: 3,  # Ready-to-eat cereals — mostly NOVA 4
    69: 3,  # Cooked cereals (plain oatmeal = NOVA 1)
    71: 3,  # Meat mixtures (sandwiches, burgers, hot dogs)
    72: 3,  # Poultry mixtures
    73: 3,  # Seafood mixtures
    74: 3,  # Egg mixtures
    75: 3,  # Meat, poultry, seafood with grains (casseroles)
    76: 3,  # Soups — mostly NOVA 3
    77: 3,  # Stews, goulash
    78: 3,  # Vegetable combinations, salads with dressing
    81: 2,  # Fats, oils (butter, olive oil = NOVA 2)
    82: 2,  # Condiments, sauces (vinegar, plain mustard = NOVA 2)
    83: 3,  # Gravies, sauces (packaged/canned = NOVA 3)
    84: 3,  # Salad dressings
    85: 3,  # Dips, spreads
    86: 3,  # Sugars, sweets, candies
    87: 3,  # Jams, jellies, preserves
    91: 4,  # Soft drinks → NOVA 4
    92: 4,  # Fruit drinks → NOVA 4
    93: 1,  # Coffee, tea (plain) → NOVA 1
    94: 4,  # Alcoholic beverages — NOVA 4 for beer/wine/spirits with additives
    95: 4,  # Energy drinks → NOVA 4
    96: 1,  # Water → NOVA 1
}

# Keywords that indicate NOVA 4 regardless of category
NOVA4_KEYWORDS = [
    r'\bsoda\b', r'\bcola\b', r'\bsprite\b', r'\bpepsi\b', r'\bcoke\b',
    r'\benergy drink\b', r'\bmonster\b', r'\bred bull\b', r'\bgatorade\b',
    r'\bpowerade\b', r'\bkool.?aid\b', r'\bjuice drink\b', r'\bfruit punch\b',
    r'\bchocolate milk\b', r'\bflavored milk\b', r'\binstant\b',
    r'\bnugget\b', r'\bnuggets\b', r'\bhotdog\b', r'\bhot dog\b',
    r'\bsausage\b', r'\bbratwurst\b', r'\bpepperoni\b', r'\bsalami\b',
    r'\bbaloney\b', r'\bbologne\b', r'\bluncheon meat\b', r'\bcold cut\b',
    r'\bfrankfurter\b', r'\bwiener\b', r'\bchicken strip\b',
    r'\bfish stick\b', r'\bfish finger\b', r'\bcorn dog\b',
    r'\binstant noodle\b', r'\bramen\b', r'\binstant soup\b',
    r'\bfrozen dinner\b', r'\bfrozen meal\b', r'\bfrozen pizza\b',
    r'\bfast food\b', r'\bburger\b', r'\bmcnugget\b',
    r'\bdonut\b', r'\bdoughnut\b', r'\bcroissant\b', r'\bdanish\b',
    r'\bpoptart\b', r'\bpop-tart\b', r'\bgranola bar\b', r'\bprotein bar\b',
    r'\bcandy\b', r'\bchocolate bar\b', r'\bm&m\b', r'\bskittles\b',
    r'\bchips\b', r'\bpretzels?\b', r'\bpopcorn\b', r'\bcheese puff\b',
    r'\bfrosting\b', r'\bicing\b', r'\bcream filling\b',
    r'\byogurt\b',  # most commercial yogurt is NOVA 4
    r'\bice cream\b', r'\bfrozen yogurt\b', r'\bsorbet\b', r'\bgelato\b',
    r'\bpudding\b', r'\bjello\b', r'\bcustard\b',
    r'\bmargarine\b', r'\bspread\b',
    r'\bketchup\b', r'\bbarbecue sauce\b', r'\bbq sauce\b',
    r'\bmayonnaise\b', r'\branch dressing\b', r'\bitalian dressing\b',
    r'\bsalad dressing\b',
    r'\bpackaged\b', r'\bprocessed\b',
    r'\bready.to.eat cereal\b', r'\bcorn flakes\b', r'\bfroot loops\b',
    r'\bfrosted\b', r'\bcocoa puff\b', r'\bhoney nut\b',
    r'\bmilk shake\b', r'\bmilkshake\b', r'\bsmoothi\b',
    r'\bwhipped cream\b', r'\bwhipped topping\b', r'\bcool whip\b',
    r'\binstant oatmeal\b',  # flavored instant = NOVA 4 vs plain = NOVA 1
    r'\bflavored oatmeal\b',
]

# Keywords that indicate NOVA 1 (override category mapping)
NOVA1_KEYWORDS = [
    r'\bplain water\b', r'\btap water\b', r'\bbottled water\b',
    r'\bblack coffee\b', r'\bplain tea\b', r'\bunsweetened tea\b',
    r'\braw\b.*\bvegetable\b', r'\braw\b.*\bfruit\b',
    r'\bfresh\b.*\bfruit\b', r'\bwhole fruit\b',
    r'\bplain oatmeal\b', r'\brolled oat\b', r'\bsteel.cut oat\b',
    r'\bboiled egg\b', r'\bscrambled egg\b', r'\bfried egg\b',
    r'\bwhole milk\b', r'\bskim milk\b', r'\b2% milk\b',
    r'\bground beef\b', r'\bground turkey\b', r'\bchicken breast\b',
    r'\bsalmon\b', r'\btuna.?steak\b', r'\bsardine\b',
    r'\bplain rice\b', r'\bcooked rice\b', r'\bbrown rice\b',
    r'\bpasta plain\b', r'\bspaghetti plain\b',
    r'\blentil\b', r'\bblack bean\b', r'\bkidney bean\b',
    r'\btofu\b', r'\btempeh\b',
    r'\bnut\b', r'\balmonds?\b', r'\bwalnuts?\b', r'\bpecans?\b',
    r'\bseed\b', r'\bsunflower seed\b', r'\bpumpkin seed\b',
]

# NOVA 2: Processed culinary ingredients
NOVA2_KEYWORDS = [
    r'\bolive oil\b', r'\bvegetable oil\b', r'\bcanola oil\b',
    r'\bbutter\b', r'\bghee\b', r'\blard\b', r'\bcoconut oil\b',
    r'\bsugar\b', r'\bbrown sugar\b', r'\bhoney\b', r'\bmaple syrup\b',
    r'\bvinegar\b', r'\bsalt\b', r'\bflour\b', r'\bcornstarch\b',
    r'\byeast\b', r'\bbaking powder\b', r'\bbaking soda\b',
]


def classify_nova(food_code: int, food_desc: str) -> int:
    """
    Classify a food item into NOVA group 1-4.
    
    Parameters
    ----------
    food_code : int
        8-digit FNDDS/WWEIA food code
    food_desc : str
        Food description string
        
    Returns
    -------
    int : NOVA group (1, 2, 3, or 4)
    """
    desc_lower = str(food_desc).lower() if food_desc else ""
    category = int(str(food_code)[:2]) if food_code else 99
    
    # Check NOVA 4 keywords first (highest priority)
    for pattern in NOVA4_KEYWORDS:
        if re.search(pattern, desc_lower):
            return 4
    
    # Check NOVA 1 keywords
    for pattern in NOVA1_KEYWORDS:
        if re.search(pattern, desc_lower):
            return 1
    
    # Check NOVA 2 keywords
    for pattern in NOVA2_KEYWORDS:
        if re.search(pattern, desc_lower):
            return 2
    
    # Special overrides for ambiguous categories
    if category == 68:  # Ready-to-eat cereals almost all NOVA 4
        return 4
    if category == 69:
        if re.search(r'plain|unflavored|unsweetened', desc_lower):
            return 1
        return 4
    if category == 93:  # Coffee/tea
        if re.search(r'plain|black|unsweetened|herbal', desc_lower):
            return 1
        if re.search(r'latte|cappuccino|frappuccino|mocha|flavored|sweetened', desc_lower):
            return 4
        return 1
    if category == 94:  # Alcohol
        if re.search(r'wine|beer|lager|ale|spirits?|liquor|whiskey|vodka|rum|tequila|gin', desc_lower):
            return 4
        return 3
    if category == 82:  # Condiments
        if re.search(r'ketchup|bbq|barbecue|mayo|ranch|caesar|thousand island', desc_lower):
            return 4
        return 2
    if category == 81:  # Fats
        if re.search(r'margarine|spread|crisco|shortening', desc_lower):
            return 4
        return 2
    if category == 14:  # Milk desserts
        if re.search(r'ice cream|frozen|pudding|milkshake', desc_lower):
            return 4
        return 3
    if category == 62:  # Breads
        if re.search(r'white bread|wonder|enriched|commercial', desc_lower):
            return 4
        if re.search(r'whole wheat|whole grain|sourdough|homemade', desc_lower):
            return 2
        return 3
    if category == 51:  # Fruits
        if re.search(r'juice|drink|punch|cocktail', desc_lower):
            return 4
        return 1
    if category == 41:  # Vegetables
        if re.search(r'sauce|gravy|stir.fry|mixed dish|casserole', desc_lower):
            return 3
        return 1
    
    # Fall back to category map
    return WWEIA_NOVA_MAP.get(category, 3)  # default = NOVA 3 if unknown


if __name__ == "__main__":
    # Quick smoke test
    tests = [
        (91101010, "Cola soda"),
        (23101000, "Ground beef, raw"),
        (96000100, "Tap water"),
        (65101000, "Chocolate chip cookie, commercial"),
        (21101000, "Scrambled egg"),
    ]
    for code, desc in tests:
        print(f"{desc}: NOVA {classify_nova(code, desc)}")
