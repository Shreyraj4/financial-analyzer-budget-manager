# Transaction categorization

Code: `backend/app/ml/categorization/` (model, evaluate, train, retrain),
`backend/app/services/categorization.py` (the decision logic in the app).
Results: `docs/metrics/categorization.json`.

## The problem

Bank narrations look like `UPI/412345678901/SWIGGY/paytm` or
`POS 4512XXXXXX1234 ZARA MUMBAI`. Goal: assign a category (Food, Transport ...).
It is **multi-class text classification**, supervised (we have labels).

## Pipeline

1. **Normalize text** (`preprocessing/text.py`): strip payment rails (UPI/POS/ACH),
   reference numbers, masked card numbers, VPA handles. Keep the merchant name.
2. **TF-IDF over character n-grams (2-5)**: turn text into numbers.
   - *TF* = how often a piece of text appears in this narration.
   - *IDF* = down-weights pieces that appear in every narration (like "UPI").
   - *Character* n-grams (not words) survive truncation ("STARBU") and share
     signal between related names ("... PHARMACY").
3. **Add amount features**: log |amount| and direction (credit/debit). Rent-sized
   debits and salary credits are hints text can't give.
4. **Logistic Regression** (multi-class): learns weights per n-gram per category,
   outputs a probability for each category.

## Why Logistic Regression and not something fancier?

Fast, strong on sparse text, well understood, gives calibrated-ish probabilities
(needed for the confidence gate), and easy to explain. LinearSVC scored similarly
but has no probabilities.

## Baselines

1. Original 8 substring rules.
2. Merchant lookup table: remember each merchant's most common category.

## Two evaluation scenarios (the key idea)

| Scenario | Split | Question answered |
|---|---|---|
| Known merchants | stratified 5-fold | Everyday case: repeat merchants |
| Unseen merchants | GroupKFold by merchant | Can it generalize to brand-new merchants? |

## Results (macro-F1)

| Model | Known | Unseen |
|---|---|---|
| Seed rules | 0.105 | 0.105 |
| Merchant lookup | 0.908 | 0.000 |
| **Logistic Regression, text + amount** | **0.999** | **0.393** |
| LinearSVC, text + amount | 0.998 | 0.316 |

Interpretation: known merchants are easy (and my generator maps each merchant to
one category, so 0.999 is optimistic). Unseen merchants are hard: nothing in the
letters of "ZARA" says "fashion". Amount features lifted unseen from 0.25 to 0.39.

## Confidence gate + human in the loop

Order: user's own past label -> curated rule -> ML if confidence >= 0.7 ->
otherwise leave blank and ask the user (show the model's guess as a suggestion).
At 0.7 confidence, accuracy on unseen merchants is ~67%. User labels are stored
(`category_source = 'user'`), remembered per merchant, and `retrain.py` folds them
back into training. This is a feedback loop / active-learning flavour.

## Concepts to be able to explain

- Multi-class vs binary classification; probabilities and `argmax`.
- TF-IDF and n-grams; why characters not words.
- Regularization (`C` in scikit-learn: smaller C = stronger regularization).
- Macro-F1 vs accuracy vs weighted-F1.
- Confidence thresholds and the coverage/accuracy trade-off.
- Why GroupKFold (leakage of merchant identity).

## Interview questions

**Q: Why character n-grams?**
A: Robust to truncation and noise, and lets the model share statistical strength
across related names.

**Q: Your known-merchant score is 0.999 - isn't that suspicious?**
A: It should make you suspicious, and I flagged it: the synthetic data maps each
merchant to one category. That's why I also evaluate unseen merchants, where it
drops to 0.39, and I report both.

**Q: How do you handle low-confidence predictions?**
A: Predictions below 0.7 are not applied; the user labels them, the label is
remembered per merchant, and later used to retrain the model.

**Q: What would improve unseen-merchant accuracy?**
A: World knowledge (e.g. an LLM classifier for merchant names), merchant
databases, or embeddings of merchant text. I chose human-in-the-loop first.

**Q: Why did the rule baseline score so low?**
A: It only knew 8 merchants. That is the point of ML here: coverage.

## Reading

- scikit-learn: Text feature extraction (TF-IDF): https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- scikit-learn: Logistic regression: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- scikit-learn tutorial: Working with text data: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
- scikit-learn: Confusion matrix and classification report: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
- StatQuest: "Logistic Regression", "Regularization (Ridge/Lasso)"
