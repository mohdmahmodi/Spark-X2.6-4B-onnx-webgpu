#!/usr/bin/env bash
# Downloads the checkpoints and evaluation data used by convert/ and tests/ into hf/, at the revisions
# the published build and its validation were made from. Needs the Hugging Face CLI (pip install huggingface_hub).
set -euo pipefail
cd "$(dirname "$0")/.."

hf download XHToken/Spark-X2.6-4B --revision d1b5f26071101c27fcd07728e7a7770e428281f4 --local-dir hf/Spark-X2.6-4B
hf download XHToken/Spark-X2.6-4B-INT4 --revision 022828e703654acef8018b2bcbeddcb5f3ac6ae6 --local-dir hf/Spark-X2.6-4B-INT4

# evaluation data (only needed for tests/)
hf download openai/gsm8k main/test-00000-of-00001.parquet --repo-type dataset \
  --revision 740312add88f781978c0658806c59bc2815b9866 --local-dir hf/data/gsm8k
hf download gorilla-llm/Berkeley-Function-Calling-Leaderboard BFCL_v3_simple.json possible_answer/BFCL_v3_simple.json \
  --repo-type dataset --revision 61fc0608cfd831fcfbbaa676ebdfef0ed963eeda --local-dir hf/data/Berkeley-Function-Calling-Leaderboard
hf download Salesforce/wikitext wikitext-103-raw-v1/test-00000-of-00001.parquet --repo-type dataset \
  --revision b08601e04326c79dfdd32d625aee71d232d685c3 --local-dir hf/data/wikitext
