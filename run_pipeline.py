import os
import re
import pandas as pd
import textstat
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RAW_PATH = "data/raw/medication_dataset.csv"
OUT_PATH = "data/processed/medication_dataset_processed.csv"

def normalize_text(x):
    return ". ".join(str(x).splitlines()).strip()

def extract_numbers(text):
    pattern = r"\d+(?:,\d{3})*(?:\.\d+)?"
    return re.findall(pattern, str(text))

def simplify_text(text):
    nums = extract_numbers(text)
    nums_str = ", ".join(nums) if nums else "NONE"

    prompt = f"""
Rewrite the following medication instruction at a 6th–8th grade reading level.

Rules:
- Keep ALL numbers exactly as written. The output must include: {nums_str}
- Keep dosage, frequency, duration, age limits, and warnings.
- Do not add new medical information.
- Use short sentences and simple words.

TEXT:
{text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()

def norm_num(n):
    return str(n).replace(",", "").strip()

def main():
    df = pd.read_csv(RAW_PATH)

    df["Original_Text"] = df["Original_Text"].apply(normalize_text)
    df["Original_Grade"] = df["Original_Text"].apply(textstat.flesch_kincaid_grade)
    df["Original_Numbers"] = df["Original_Text"].apply(extract_numbers)

    if "Simplified_Text" not in df.columns:
        df["Simplified_Text"] = pd.NA

    missing = df["Simplified_Text"].isna()
    df.loc[missing, "Simplified_Text"] = df.loc[missing, "Original_Text"].apply(simplify_text)

    df["Simplified_Grade"] = df["Simplified_Text"].apply(textstat.flesch_kincaid_grade)
    df["Simplified_Numbers"] = df["Simplified_Text"].apply(extract_numbers)

    df["Numbers_Lost"] = df.apply(
        lambda r: sorted(list(set(map(norm_num, r["Original_Numbers"])) -
                              set(map(norm_num, r["Simplified_Numbers"])))),
        axis=1
    )

    df["Numbers_OK"] = df["Numbers_Lost"].apply(lambda x: len(x) == 0)
    df["Grade_Change"] = df["Original_Grade"] - df["Simplified_Grade"]

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("Saved to:", OUT_PATH)
    print("Average Original Grade:", df["Original_Grade"].mean())
    print("Average Simplified Grade:", df["Simplified_Grade"].mean())
    print("Safety %:", df["Numbers_OK"].mean() * 100)

if __name__ == "__main__":
    main()