# Semantic match (PENDULUM)

Read queue.json and requests/ only. Each request gives a question, the answer extracted from one reply, and two
reference answers A and B (their order is random). Write labels/<key>.json:
{"match": "A" | "B" | "both" | "neither", "why": "<one short sentence>"}
"A" = the extracted answer means the same thing as reference A for this question (synonyms, singular/plural,
more specific or general names of the same thing count as the same when the question's level of detail allows);
"both" only if it cannot be distinguished; "neither" otherwise.
