import os
from dotenv import load_dotenv

from chains.extract_chain import extract_chain
from chains.match_chain import match_chain
from chains.score_chain import score_chain
from chains.explain_chain import explain_chain

load_dotenv()

def read_file(path):
    with open(path, "r") as f:
        return f.read()

job_desc = read_file("data/job_description.txt")

resumes = {
    "Strong": read_file("data/resumes/strong.txt"),
    "Average": read_file("data/resumes/average.txt"),
    "Weak": read_file("data/resumes/weak.txt")
}

for name, resume in resumes.items():
    print(f"\n--- {name} Candidate ---")

    extracted = extract_chain.invoke({"resume": resume})
    matched = match_chain.invoke({
        "resume_data": extracted,
        "job_description": job_desc
    })
    score = score_chain.invoke({"match_data": matched})
    explanation = explain_chain.invoke({
        "score": score,
        "match_data": matched
    })

    print("Score:", score)
    print("Explanation:", explanation)