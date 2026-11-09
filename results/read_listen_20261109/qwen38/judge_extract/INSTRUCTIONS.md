# Answer extraction (blind)

Read queue.json and the files in requests/ only. Do not read private_keymap.json, records, reports or any gold
labels. Each request gives a question about an image, the image path, and one model's complete final reply
(and, only if the final reply is empty, the end of its reasoning). You are not told which model, method or
condition produced the reply.

For each key, view the image if needed, read the complete reply, and write labels/<key>.json:

- PathVQA (yes/no question): {"answer": "yes" | "no" | "other" | "ambiguous" | "no_answer", "quote": "<verbatim
  span from the reply that states the answer>"}. Extract the reply's final answer to the original question, not
  whether it agrees with anyone. "ambiguous" = the reply commits to both or hedges without a final answer;
  "no_answer" = empty or truncated before any answer.
- PENDULUM (open question): {"answer": "<the reply's final answer as a short phrase, as stated>" or "no_answer",
  "quote": "<verbatim span>"}.

The quote must be copied exactly from the reply text given in the request. Judge each key independently.
