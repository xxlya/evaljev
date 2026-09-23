"""A support assistant with three Jev decision points, instrumented end to end.

This is the shape most deployed Jev workflows have: a request arrives, a few typed
decisions route it, and code acts on the result. It exists to be copied — swap the
questions and the policy for yours and the dashboard works the same way.

    python examples/support_assistant.py --phase v1 --out examples/support-v1.jsonl
    python examples/support_assistant.py --phase v2 --out examples/support-v2.jsonl
    python examples/support_assistant.py --phase v3 --out examples/support-v3.jsonl
    python examples/support_assistant.py --phase v4 --out examples/support-v4.jsonl
    evaljev report examples/support-v*.jsonl -o report.html

The three phases are the same workflow with two edits to one question in between —
the kind nobody writes a test for:

- ``v1`` the original.
- ``v2`` a new product line ships, so someone adds a ``billing`` option. Worth
  watching even so: a new option changes what every ``confidence`` threshold
  means, because Jev reports confidence as a margin over ``1/k``.
- ``v3`` someone tidies the question up — shorter instructions, shorter option
  descriptions. Nothing is renamed and nothing is removed.
- ``v4`` the labels get sorted alphabetically in one file while their descriptions
  stay in insertion order in another, so every description ends up on the wrong
  label. Same six labels, same ``k``, same instructions.

Which of those three actually hurt is not obvious in advance, and is the whole
point of recording the traces.

Needs JEV_API_KEY. Roughly $0.015 a phase at three decisions per request.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from evaljev import (
    BudgetedJevClient,
    BudgetExceeded,
    JevHTTPClient,
    SpendBudget,
    answer_branch,
    watch,
)

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = "support-assistant"

# The probability below which the assistant stops and asks a person. A probability,
# not the API's `confidence`, which rescales with the number of options — the exact
# trap phase v2 walks into.
REVIEW_BELOW = 0.60

CATEGORIES_V1 = {
    "refund": "The customer wants money back for something already paid.",
    "order_status": "The customer wants to know where an order is or when it arrives.",
    "cancel_subscription": "The customer wants a recurring plan stopped.",
    "change_address": "The customer wants a delivery address corrected.",
    "other": "None of the above is being requested.",
}

# Phase two. A teammate ships a billing product line and adds one option. Nothing
# else changes: same instructions, same model, same policy, same five old labels.
CATEGORIES_V2 = {
    **CATEGORIES_V1,
    "billing": "The customer has a question about a charge, an invoice or a payment.",
}

# Phase three. Someone shortens the schema — the descriptions get terser and the
# broad billing bucket now overlaps refund, while the instruction that disambiguated
# them ("judge only what is requested") is gone. Same six labels, same model, same
# policy, same customers: a diff that looks like tidying up.
CATEGORIES_V3 = {
    "refund": "Money back.",
    "order_status": "Delivery updates.",
    "cancel_subscription": "Stop a plan.",
    "change_address": "Address changes.",
    "billing": "Anything about a charge, an invoice or a payment.",
    "other": "Anything else.",
}
INSTRUCTIONS_V3 = "Pick the best category for this message."

# Phase four. The labels and their descriptions live in two places, and someone
# sorts the labels alphabetically for readability without touching the other one.
# zip() quietly re-pairs them: six labels, six descriptions, every description now
# belonging to a different category. Same keys, same k, same instructions — the
# schema validates, the answers stay inside the option set, and nothing in a test
# suite compares a label against its own description.
_LABELS_SORTED_FOR_READABILITY = sorted(CATEGORIES_V2)
_DESCRIPTIONS_STILL_IN_INSERTION_ORDER = list(CATEGORIES_V2.values())
CATEGORIES_V4 = dict(
    zip(_LABELS_SORTED_FOR_READABILITY, _DESCRIPTIONS_STILL_IN_INSERTION_ORDER)
)


INSTRUCTIONS_V1 = (
    "Which single action is this customer asking for? Judge only what is "
    "requested, not what is mentioned in passing."
)


def classify_question(categories: dict, instructions: str = INSTRUCTIONS_V1) -> dict:
    return {
        "category": {
            "type": "choice",
            "instructions": instructions,
            "criteria": categories,
        }
    }


PHASES = {
    "v1": (CATEGORIES_V1, INSTRUCTIONS_V1, "category-v1"),
    "v2": (CATEGORIES_V2, INSTRUCTIONS_V1, "category-v2"),
    "v3": (CATEGORIES_V3, INSTRUCTIONS_V3, "category-v3"),
    "v4": (CATEGORIES_V4, INSTRUCTIONS_V1, "category-v4"),
}


URGENCY_QUESTION = {
    "urgency": {
        "type": "score",
        "instructions": "How quickly does this need a human, given only what the "
        "customer wrote?",
        "criteria": [
            "Routine — can wait for the normal queue.",
            "Time-sensitive — a deadline or a delivery date is at stake.",
            "Blocking — the customer cannot use what they paid for.",
            "Escalation — money already lost, or an explicit complaint about the service.",
        ],
    }
}

HANDOFF_QUESTION = {
    "needs_human": {
        "type": "noul",
        "instructions": "Does answering this correctly require a person, because it "
        "needs account access, a judgement call, or an exception to policy?",
    }
}

# Realistic support messages written for the demo. The messages are made up; every
# probability, latency and cost recorded against them is a real API response.
REQUESTS: list[tuple[str, str]] = [
    ("Where is my parcel? It was due Tuesday and nothing has arrived.", "order_status"),
    ("I was charged twice for order 4471 — please send the second one back.", "refund"),
    ("Please stop my monthly plan, I don't use it any more.", "cancel_subscription"),
    ("Do you ship to Norway?", "other"),
    ("Tracking says delivered but there's nothing at my door.", "order_status"),
    ("You took £59 out twice in March. I want the duplicate back.", "refund"),
    ("Moving on Friday — can you send it to 42 Kestrel Lane instead?", "change_address"),
    ("Any update on order 5120? It's been a week.", "order_status"),
    ("The jacket arrived torn. I'd like my money back rather than a replacement.", "refund"),
    ("Is the store open on Sunday?", "other"),
    ("Cancel the subscription before it renews on the 3rd please.", "cancel_subscription"),
    ("Order 8830 has been 'in transit' for nine days.", "order_status"),
    ("I want my money back for the annual plan I never used.", "refund"),
    ("Wrong flat number on my order — it should be 12B not 12.", "change_address"),
    ("What material is the blue rug made of?", "other"),
    ("My order was supposed to arrive before my daughter's birthday, it's tomorrow.", "order_status"),
    ("I was billed after cancelling. Send that payment back please.", "refund"),
    ("Please close my account and stop billing me.", "cancel_subscription"),
    ("Has my replacement shipped yet?", "order_status"),
    ("Do you have the lamp in black?", "other"),
    ("Send back the delivery fee — it arrived four days late.", "refund"),
    ("New address: 9 Harbour Road, please update before dispatch.", "change_address"),
    ("Still nothing on order 2290, the tracking hasn't moved since Monday.", "order_status"),
    ("Can I talk to someone about a bulk order?", "other"),
    ("There's a £12 charge on my card I did not authorise — return it.", "refund"),
    ("Turn off auto-renew.", "cancel_subscription"),
    ("When will order 6701 leave the warehouse?", "order_status"),
    ("I typed the postcode wrong, it's SW1A 2AA.", "change_address"),
    ("Your checkout page kept freezing last night.", "other"),
    ("The shoes never came and the money has gone. I want it refunded.", "refund"),
    ("Expected delivery said Thursday, it's Saturday now.", "order_status"),
    ("End my membership at the end of this billing period.", "cancel_subscription"),
    ("Can you redirect today's delivery to my office at 5 Mill Street?", "change_address"),
    ("Is there a warranty on the desk lamp?", "other"),
    ("Refund the express shipping I paid for — it came by normal post.", "refund"),
    ("Parcel 3341 says 'awaiting collection' but nobody collected it.", "order_status"),
    ("I'd like to stop the plan but keep my saved designs.", "cancel_subscription"),
    ("Which courier do you use for large items?", "other"),
    ("My invoice shows two line items for the same order — I want one back.", "refund"),
    ("Deliver to the neighbour at number 7 instead of me.", "change_address"),
    ("It's been ten days and order 9120 has not shipped.", "order_status"),
    ("How long does assembly usually take?", "other"),
    ("The subscription renewed after I cancelled. Give me that money back.", "refund"),
    ("Where has my order got to? Reference 7781.", "order_status"),
    ("Stop my plan, I'm switching to a competitor.", "cancel_subscription"),
]

# The later phases see the next day's traffic: the same customers, the same mix.
# Most messages repeat, which is what makes a before/after comparison of the
# identical input possible.
PHASE_REPEAT = list(range(24))


def run(phase: str, out: Path, cap: float) -> int:
    categories, instructions, version = PHASES[phase]
    questions = classify_question(categories, instructions)
    requests = REQUESTS if phase == "v1" else [REQUESTS[i] for i in PHASE_REPEAT]

    out.unlink(missing_ok=True)
    budget = SpendBudget({"jev": cap})
    # One line: the client records every decision it makes. Nothing below changes
    # shape because of it — `client.decide(...)` is the call this workflow already had.
    client = watch(
        BudgetedJevClient(JevHTTPClient(), budget),
        workflow_id=WORKFLOW,
        path=out,
        workflow_version="assistant-3",
    )

    handled, escalated = 0, 0
    for i, (message, intended) in enumerate(requests, 1):
        request_id = f"{phase}-{uuid.uuid4().hex[:8]}"

        def review_policy(answers, _name="category"):
            answer = answers[_name]
            probs = answer.probabilities or {}
            p_max = max(probs.values()) if probs else 1.0
            return "needs_review" if p_max < REVIEW_BELOW else answer.selected

        try:
            # Everything inside this block belongs to one request, so the console can
            # show the path it took rather than three unrelated decisions.
            with client.request(request_id, phase=phase, intended=intended):
                with client.step(
                    "classify_request",
                    policy=review_policy,
                    question_version=version,
                    policy_version=f"review-below-{REVIEW_BELOW}",
                ):
                    client.decide(
                        state={"message": message, "channel": "email"}, questions=questions
                    )
                category_trace = client.last_trace
                category = category_trace.action

                with client.step(
                    "rate_urgency",
                    # answer_branch, not int(value): a score answer's `value` is the
                    # expected level, and truncating it can land on a level the model
                    # gave no probability to at all.
                    policy=lambda answers: f"priority_{answer_branch(answers['urgency'])}",
                    question_version="urgency-v1",
                    policy_version="argmax-level",
                ):
                    # Later steps see what the earlier ones decided — which is why a
                    # failure at the first step does not stay at the first step.
                    client.decide(
                        state={"message": message, "category": category},
                        questions=URGENCY_QUESTION,
                    )
                urgency_trace = client.last_trace

                with client.step(
                    "route_handoff",
                    policy=lambda answers: (
                        "human" if (answers["needs_human"].value or 0) >= 0.5 else "auto"
                    ),
                    question_version="handoff-v1",
                    policy_version="noul-at-0.5",
                ):
                    client.decide(
                        state={
                            "message": message,
                            "category": category,
                            "urgency": urgency_trace.action,
                        },
                        questions=HANDOFF_QUESTION,
                    )
                handoff_trace = client.last_trace
        except BudgetExceeded as exc:
            print(f"  stopped after {i - 1} requests: {exc}")
            break

        to_human = category == "needs_review" or handoff_trace.action == "human"
        escalated += to_human
        handled += 1

        # A reviewer only labels what they actually look at, so outcomes come back
        # for a subset — the normal case, and the reason the label-free signals are
        # the ones that have to work.
        if i % 3 == 0:
            client.record_outcome(
                {"reviewer_category": intended, "routed_to": category},
                correct=category == intended,
                trace=category_trace,
            )
            client.record_outcome(
                {"went_to_human": to_human},
                correct=None if to_human else True,
                trace=handoff_trace,
            )

        print(
            f"  {i:>2}/{len(requests)}  {category:<18} {urgency_trace.action:<11}"
            f"{'HUMAN' if to_human else 'auto':<6} ${budget.spent['jev']:.4f}"
        )

    print(
        f"\n{handled} requests · {escalated} sent to a human "
        f"({escalated / handled:.0%}) · ${budget.spent['jev']:.4f}"
    )
    print(f"wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=sorted(PHASES), default="v1")
    parser.add_argument("--out", default=None)
    parser.add_argument("--cap", type=float, default=1.00, help="Jev USD ceiling")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env", override=False)
    except ImportError:
        pass

    out = Path(args.out or REPO / "examples" / f"support-{args.phase}.jsonl")
    return run(args.phase, out, args.cap)


if __name__ == "__main__":
    raise SystemExit(main())
