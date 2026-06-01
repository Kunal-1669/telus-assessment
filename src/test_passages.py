"""Test passages for paragraph-level paraphrase evaluation.

Contains the assessment's cover letter sample plus two complex domain
passages (legal contract breach, type-2 diabetes management) each in the
200-400 word range. Imported by bench_cpu, evaluation, and demo scripts.
"""
from __future__ import annotations

from pathlib import Path


PASSAGES: dict[str, dict] = {
    "cover_letter": {
        "domain": "career",
        "text": (
            "A cover letter is a formal document that accompanies your resume "
            "when you apply for a job. It serves as an introduction and "
            "provides additional context for your application. Here's a "
            "breakdown of its various aspects: Purpose: The primary purpose "
            "of a cover letter is to introduce yourself to the hiring manager "
            "and to provide context for your resume. It allows you to "
            "elaborate on your qualifications, skills, and experiences in a "
            "way that your resume may not fully capture. It's also an "
            "opportunity to express your enthusiasm for the role and the "
            "company, and to explain why you would be a good fit. Content: "
            "A typical cover letter includes the following sections: Header: "
            "Includes your contact information, the date, and the employer's "
            "contact information. Salutation: A greeting to the hiring "
            "manager, preferably personalised with their name. Introduction: "
            "Briefly introduces who you are and the position you're applying "
            "for. Body: This is the core of your cover letter where you "
            "discuss your qualifications, experiences, and skills that make "
            "you suitable for the job. You can also mention how you can "
            "contribute to the company. Conclusion: Summarises your points "
            "and reiterates your enthusiasm for the role. You can also "
            "include a call to action, like asking for an interview. "
            "Signature: A polite closing followed by your name. The cover "
            "letter is often the first document that a hiring manager will "
            "read, so it sets the tone for your entire application. It "
            "provides you with a chance to stand out among other applicants "
            "and to make a strong first impression."
        ),
    },

    "legal_contract_breach": {
        "domain": "legal",
        "text": (
            "A material breach of contract occurs when one party fails to "
            "fulfill an obligation that goes to the essence of the "
            "agreement, thereby depriving the non-breaching party of the "
            "benefit it reasonably expected to receive. Under common law "
            "jurisdictions, the doctrine of substantial performance "
            "recognizes that minor deviations from contract terms do not "
            "necessarily entitle the other party to terminate the contract, "
            "provided the essential purpose of the agreement has been "
            "substantially achieved. However, when a breach is deemed "
            "material, the aggrieved party may pursue several remedies, "
            "including compensatory damages, specific performance, or "
            "rescission of the contract. Compensatory damages aim to place "
            "the non-breaching party in the position they would have "
            "occupied had the contract been fully performed, and may "
            "include both direct losses and consequential damages that "
            "were reasonably foreseeable at the time of contracting. "
            "Specific performance, an equitable remedy, is typically "
            "reserved for situations where monetary compensation would be "
            "inadequate, such as in contracts involving unique goods or "
            "real property. Courts will generally not order specific "
            "performance for personal service contracts due to the "
            "impracticality of judicial supervision. The doctrine of "
            "mitigation requires the non-breaching party to take reasonable "
            "steps to minimize their losses; failure to mitigate may reduce "
            "the damages recoverable. In commercial contexts, parties "
            "frequently include liquidated damages clauses to predetermine "
            "the amount of compensation payable upon breach, although such "
            "clauses must represent a genuine pre-estimate of loss and not "
            "constitute a penalty, as penalty clauses are generally "
            "unenforceable. Statutes of limitations impose time limits "
            "within which a claim must be brought, and these vary by "
            "jurisdiction and the nature of the contract."
        ),
    },

    "medical_diabetes": {
        "domain": "medical",
        "text": (
            "Type 2 diabetes mellitus is a chronic metabolic disorder "
            "characterized by insulin resistance and progressive beta-cell "
            "dysfunction, resulting in elevated blood glucose levels that, "
            "if poorly controlled, can lead to a range of microvascular and "
            "macrovascular complications. The pathophysiology involves a "
            "complex interplay between genetic predisposition, lifestyle "
            "factors, and environmental influences, with obesity, physical "
            "inactivity, and dietary patterns being particularly important "
            "contributors to disease onset. Initial management typically "
            "emphasizes lifestyle modifications, including dietary changes, "
            "increased physical activity, weight reduction, and smoking "
            "cessation, as these interventions can substantially improve "
            "glycemic control and reduce cardiovascular risk. When lifestyle "
            "measures alone fail to achieve target glycated hemoglobin "
            "levels, pharmacological therapy becomes necessary. Metformin "
            "remains the first-line agent in most clinical guidelines due "
            "to its efficacy, favorable safety profile, and beneficial "
            "effects on cardiovascular outcomes. For patients who do not "
            "achieve glycemic targets with metformin alone, treatment "
            "intensification may involve the addition of agents such as "
            "sulfonylureas, dipeptidyl peptidase-4 inhibitors, "
            "sodium-glucose cotransporter-2 inhibitors, glucagon-like "
            "peptide-1 receptor agonists, or insulin. The choice of "
            "additional therapy should be individualized based on "
            "patient-specific factors including comorbidities, the risk of "
            "hypoglycemia, weight considerations, and cost. Regular "
            "monitoring of glycated hemoglobin every three to six months "
            "is essential to evaluate treatment efficacy and guide "
            "therapeutic adjustments. Patients should also undergo periodic "
            "screening for diabetic complications, including retinopathy, "
            "nephropathy, and neuropathy. Patient education regarding "
            "self-monitoring of blood glucose, recognition of hypoglycemia, "
            "and adherence to medication regimens is fundamental to "
            "achieving optimal long-term outcomes."
        ),
    },
}


def get(passage_id: str) -> dict:
    if passage_id not in PASSAGES:
        raise KeyError(f"Unknown passage id: {passage_id!r}. "
                       f"Available: {list(PASSAGES)}")
    return PASSAGES[passage_id]


def list_passages() -> list[tuple[str, str, int]]:
    """Returns [(id, domain, word_count), ...]"""
    return [
        (pid, p["domain"], len(p["text"].split()))
        for pid, p in PASSAGES.items()
    ]


def dump_to_files(out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for pid, p in PASSAGES.items():
        path = out_dir / f"{pid}.txt"
        path.write_text(p["text"] + "\n")
        written.append(path)
    return written


def main():
    import argparse
    p = argparse.ArgumentParser(description="List or dump test passages")
    p.add_argument("--dump-to", help="Write each passage to a .txt file in this dir")
    p.add_argument("--show", help="Print the named passage to stdout")
    args = p.parse_args()

    if args.show:
        print(get(args.show)["text"])
        return

    print(f"{'id':<25} {'domain':<10} {'words':>6}")
    print("-" * 45)
    for pid, domain, wc in list_passages():
        print(f"{pid:<25} {domain:<10} {wc:>6}")

    if args.dump_to:
        written = dump_to_files(args.dump_to)
        print(f"\nWrote {len(written)} files to {args.dump_to}/")


if __name__ == "__main__":
    main()
