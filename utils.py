import os
import re
from dotenv import load_dotenv
from openai import OpenAI
# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def normalize_text(x):
    """Normalize whitespace for consistent grading."""
    return ". ".join(str(x).splitlines()).strip()


def digit_tokens(text):
    """
    Return whitespace-separated tokens containing digits.
    This will help capture values attached to units or ranges, such as:
    5mg, 10ML, 100-125, 45%, 0.5, mL/min/1.73
    """
    if not isinstance(text, str):
        return []
    
    exclude_prefixes = (
        "beta", "cyp", "p450", "co2", "oatp", "hba1c"   
    )

    keep = []
    for token in str(text).split():
        tok = str(token).strip()

        # must contain a digit
        if not any(ch.isdigit() for ch in tok):
            continue

        low = tok.lower()
        if (low.startswith(exclude_prefixes)):
            continue
        keep.append(token)
        
    return [token for token in keep if any(ch.isdigit() for ch in token)]

def normalize_digit_token(token):
    """
    Normalize full digit-containing tokens for safe comparison
    Keep units/ranges attached, but remove outside punctuation.
    """
    token = str(token).strip()
    token = token.strip("()[]{};:,\"'")

    #remove commas inside numbers
    token = re.sub(r"(?<=\d),(?=\d)", "", token)

    # normalize .5-> 0.5 when token begins with decimal
    if token.startswith("."):
        token = "0" + token
    
    # normalize long dashes
    token = token.replace("—", "-").replace("–", "-")
    
    return token

def get_missing_tokens(original_text, simplified_text):
    orig_tokens = [normalize_digit_token(t) for t in digit_tokens(original_text)]
    simp_tokens = [normalize_digit_token(t) for t in digit_tokens(simplified_text)]

    simp_set = set(simp_tokens)
    return [tok for tok in orig_tokens if tok not in simp_set]


#----------------------------
#Main simplification function
#----------------------------

def simplify_text(text, max_attempts=3):
    """
    Simplify medication instructions using GPT-4o while preserving numeric dosage and safety information. 
    """
    original = str(text)

    prompt = f"""
Rewrite the following medication instructions at a STRICT 7th grade reading level.
Formatting Requirements:
Output format (must follow):
- Starts with: "How to take it:"
- Use bullet points
- Each bullet point should be a short, clear sentence.(max 8-10 words)
- If sentence has 'and', or 'or', split into two bullet points.
- Do not use semicolons.

- Avoid complex medical jargon. 
-Use simple words such as:
    -"take" instead of "administer" 
    -"medicine" instead of "medication" 
    -"doctor" instead of "physician" 
    -"side effects" instead of "adverse reactions"
-Do not add new medical information

Simplify the following medical instructions so they are easier for patients to understand.
Rules:
- Keep all numbers same do not remove or change them.
- Keep dosage amounts and time durations unchanged
- Only simplify the wording around them.
- Keep every numerical token EXACTLY as it appears.
- Do not add commas(2000, stays 2000, not 2,000)
- Do not change ranges (100-125 must stay 100-125)
- Do Not change decimals (0.5 stays 0.5)
- Keep ranges, dosages, frequencies, durations, age limits, and warnings.
- Do NOT remove, round, summarize, or alter any numbers in any way.
- Do not change units formatting (mL/min/1.73 stays mL/min/1.73).
- Do not spell out numbers. Keep numbers written as digits. Example: 3 months should stay 3 months.
- Keep percentage EXACTLY the same characters( 45% stays 45%, not 45 percent or 45 %).

Before Responding:
1. Identify every numeric token in the original text.
2. Rewrite the text.
3. Verify every numeric token appears exactly in output.
4. If any number is missing, fix it before finishing.
5. Only output the final rewritten text.
TEXT:
{original}
""".strip()
    
    best_output = original
    best_missing = None

    for _ in range(max_attempts):
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        simplified = resp.choices[0].message.content.strip()
        missing = get_missing_tokens(original, simplified)

        if not missing:
            return simplified
        
        if best_missing is None or len(missing) < len(best_missing):
            best_output = simplified
            best_missing = missing

    return best_output

