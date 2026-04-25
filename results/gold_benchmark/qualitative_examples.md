# Qualitative Error Analysis: gold_bert_ner

**Total Examples:** 30

---

## Example 1: ❌ ERROR

**Query:** `bullfrog kicker bf400`

**Error Types:** FP, FN

| Token | Gold | Predicted |
|-------|------|-----------|
| bullfrog | O | B-PRODUCT_TYPE ⚠️ |
| kicker | B-BRAND | O ⚠️ |
| bf400 | O | O |

---

## Example 2: ❌ ERROR

**Query:** `paint by numbers for adults realistic owls`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| paint | O | O |
| by | O | O |
| numbers | O | O |
| for | O | O |
| adults | O | O |
| realistic | O | O |
| owls | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 3: ❌ ERROR

**Query:** `dragon ball z dvd season 1`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| dragon | O | B-BRAND ⚠️ |
| ball | O | I-PRODUCT_TYPE ⚠️ |
| z | O | I-PRODUCT_TYPE ⚠️ |
| dvd | O | O |
| season | O | B-PRODUCT_TYPE ⚠️ |
| 1 | O | I-PRODUCT_TYPE ⚠️ |

---

## Example 4: ❌ ERROR

**Query:** `peter ustinov`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| peter | O | B-PRODUCT_TYPE ⚠️ |
| ustinov | O | O |

---

## Example 5: ❌ ERROR

**Query:** `10k gold ring without stones`

**Error Types:** FP, FN

| Token | Gold | Predicted |
|-------|------|-----------|
| 10k | O | O |
| gold | B-COLOR | O ⚠️ |
| ring | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| without | O | O |
| stones | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 6: ❌ ERROR

**Query:** `posters of harley davidson motorcycles`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| posters | O | B-PRODUCT_TYPE ⚠️ |
| of | O | O |
| harley | O | O |
| davidson | O | O |
| motorcycles | O | O |

---

## Example 7: ❌ ERROR

**Query:** `henry fuller`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| henry | O | O |
| fuller | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 8: ❌ ERROR

**Query:** `jordan jersey`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| jordan | O | B-PRODUCT_TYPE ⚠️ |
| jersey | O | O |

---

## Example 9: ❌ ERROR

**Query:** `lion head bracelet for men`

**Error Types:** FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| lion | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| head | I-PRODUCT_TYPE | B-PRODUCT_TYPE ⚠️ |
| bracelet | I-PRODUCT_TYPE | B-PRODUCT_TYPE ⚠️ |
| for | B-PRODUCT_TYPE | O ⚠️ |
| men | I-PRODUCT_TYPE | O ⚠️ |

---

## Example 10: ❌ ERROR

**Query:** `yeti backpack cooler`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| yeti | B-BRAND | O ⚠️ |
| backpack | B-PRODUCT_TYPE | O ⚠️ |
| cooler | O | O |

---

## Example 11: ❌ ERROR

**Query:** `nivea tinted moisturizer`

**Error Types:** FP, FN

| Token | Gold | Predicted |
|-------|------|-----------|
| nivea | B-BRAND | O ⚠️ |
| tinted | O | O |
| moisturizer | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 12: ❌ ERROR

**Query:** `silver tablecloth 120`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| silver | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| tablecloth | B-PRODUCT_TYPE | O ⚠️ |
| 120 | O | O |

---

## Example 13: ❌ ERROR

**Query:** `double tab aerial hoop for sale aerials usa`

**Error Types:** FP, FN

| Token | Gold | Predicted |
|-------|------|-----------|
| double | O | O |
| tab | O | B-PRODUCT_TYPE ⚠️ |
| aerial | O | O |
| hoop | O | O |
| for | O | O |
| sale | O | O |
| aerials | O | O |
| usa | B-PRODUCT_TYPE | O ⚠️ |

---

## Example 14: ❌ ERROR

**Query:** `10k gold ring without stones`

**Error Types:** FP, FN

| Token | Gold | Predicted |
|-------|------|-----------|
| 10k | O | O |
| gold | B-COLOR | O ⚠️ |
| ring | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| without | O | O |
| stones | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 15: ❌ ERROR

**Query:** `classic van s men old school`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| classic | O | O |
| van | B-BRAND | O ⚠️ |
| s | B-PRODUCT_TYPE | O ⚠️ |
| men | I-PRODUCT_TYPE | O ⚠️ |
| old | O | O |
| school | O | O |

---

## Example 16: ❌ ERROR

**Query:** `side post battery for a`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| side | B-COLOR | O ⚠️ |
| post | I-COLOR | O ⚠️ |
| battery | B-PRODUCT_TYPE | O ⚠️ |
| for | O | O |
| a | O | O |

---

## Example 17: ❌ ERROR

**Query:** `jewelry stainless steel`

**Error Types:** TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| jewelry | O | O |
| stainless | B-MATERIAL | B-PRODUCT_TYPE ⚠️ |
| steel | I-MATERIAL | B-PRODUCT_TYPE ⚠️ |

---

## Example 18: ❌ ERROR

**Query:** `vintage sewing machine music box`

**Error Types:** TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| vintage | O | O |
| sewing | O | O |
| machine | B-PRODUCT_TYPE | I-PRODUCT_TYPE ⚠️ |
| music | O | O |
| box | O | O |

---

## Example 19: ❌ ERROR

**Query:** `oak hill pants big and tall`

**Error Types:** FP, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| oak | B-BRAND | B-BRAND |
| hill | I-BRAND | B-PRODUCT_TYPE ⚠️ |
| pants | O | B-PRODUCT_TYPE ⚠️ |
| big | O | O |
| and | O | O |
| tall | O | O |

---

## Example 20: ❌ ERROR

**Query:** `lime green twin fitted sheet`

**Error Types:** FP, FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| lime | B-PRODUCT_TYPE | B-BRAND ⚠️ |
| green | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| twin | B-PRODUCT_TYPE | O ⚠️ |
| fitted | B-PRODUCT_TYPE | O ⚠️ |
| sheet | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 21: ❌ ERROR

**Query:** `kresley cole immortals after dark`

**Error Types:** FP, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| kresley | O | B-BRAND ⚠️ |
| cole | O | O |
| immortals | O | O |
| after | O | O |
| dark | B-PRODUCT_TYPE | I-PRODUCT_TYPE ⚠️ |

---

## Example 22: ❌ ERROR

**Query:** `white shirts for men`

**Error Types:** FP, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| white | B-COLOR | B-PRODUCT_TYPE ⚠️ |
| shirts | O | B-PRODUCT_TYPE ⚠️ |
| for | O | O |
| men | O | O |

---

## Example 23: ❌ ERROR

**Query:** `small ivory lamp`

**Error Types:** TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| small | O | O |
| ivory | B-PRODUCT_TYPE | B-MATERIAL ⚠️ |
| lamp | O | O |

---

## Example 24: ❌ ERROR

**Query:** `end of evangelion`

**Error Types:** FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| end | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| of | I-PRODUCT_TYPE | B-PRODUCT_TYPE ⚠️ |
| evangelion | I-PRODUCT_TYPE | O ⚠️ |

---

## Example 25: ✅ CORRECT

**Query:** `mens hoodies under 10`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| mens | O | O |
| hoodies | O | O |
| under | O | O |
| 10 | O | O |

---

## Example 26: ✅ CORRECT

**Query:** `closet storage`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| closet | O | O |
| storage | O | O |

---

## Example 27: ✅ CORRECT

**Query:** `head phones`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| head | O | O |
| phones | O | O |

---

## Example 28: ✅ CORRECT

**Query:** `open ended toys for 1 year old`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| open | O | O |
| ended | O | O |
| toys | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| for | O | O |
| 1 | O | O |
| year | O | O |
| old | B-PRODUCT_TYPE | B-PRODUCT_TYPE |

---

## Example 29: ✅ CORRECT

**Query:** `chameleon substrate`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| chameleon | O | O |
| substrate | O | O |

---

## Example 30: ✅ CORRECT

**Query:** `library organization`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| library | O | O |
| organization | O | O |

---
