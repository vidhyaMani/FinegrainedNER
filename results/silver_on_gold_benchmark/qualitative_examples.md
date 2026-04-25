# Qualitative Error Analysis: roberta_ner

**Total Examples:** 30

---

## Example 1: ❌ ERROR

**Query:** `waist beads`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| waist | O | O |
| beads | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 2: ❌ ERROR

**Query:** `watch for falling rocks`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| watch | O | B-PRODUCT_TYPE ⚠️ |
| for | O | O |
| falling | O | O |
| rocks | O | O |

---

## Example 3: ❌ ERROR

**Query:** `humanrace shoes men`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| humanrace | O | O |
| shoes | O | B-PRODUCT_TYPE ⚠️ |
| men | O | O |

---

## Example 4: ❌ ERROR

**Query:** `solid color iphone 8 plus case`

**Error Types:** FP, FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| solid | O | O |
| color | O | B-PRODUCT_TYPE ⚠️ |
| iphone | O | O |
| 8 | B-PRODUCT_TYPE | O ⚠️ |
| plus | I-PRODUCT_TYPE | B-PRODUCT_TYPE ⚠️ |
| case | O | O |

---

## Example 5: ❌ ERROR

**Query:** `adidas trefoil shirts for men`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| adidas | O | B-BRAND ⚠️ |
| trefoil | O | O |
| shirts | O | O |
| for | O | O |
| men | O | O |

---

## Example 6: ❌ ERROR

**Query:** `charmed poster`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| charmed | O | O |
| poster | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 7: ❌ ERROR

**Query:** `double tab aerial hoop for sale aerials usa`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| double | O | O |
| tab | O | O |
| aerial | O | O |
| hoop | O | B-PRODUCT_TYPE ⚠️ |
| for | O | O |
| sale | O | O |
| aerials | O | O |
| usa | B-PRODUCT_TYPE | B-PRODUCT_TYPE |

---

## Example 8: ❌ ERROR

**Query:** `the cars cd boxset`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| the | O | O |
| cars | O | O |
| cd | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| boxset | O | I-PRODUCT_TYPE ⚠️ |

---

## Example 9: ❌ ERROR

**Query:** `absorbent underwear for women`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| absorbent | O | O |
| underwear | O | O |
| for | O | O |
| women | B-PRODUCT_TYPE | O ⚠️ |

---

## Example 10: ❌ ERROR

**Query:** `lion head bracelet for men`

**Error Types:** FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| lion | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| head | I-PRODUCT_TYPE | I-PRODUCT_TYPE |
| bracelet | I-PRODUCT_TYPE | B-PRODUCT_TYPE ⚠️ |
| for | B-PRODUCT_TYPE | O ⚠️ |
| men | I-PRODUCT_TYPE | O ⚠️ |

---

## Example 11: ❌ ERROR

**Query:** `police shirt women`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| police | B-COLOR | B-COLOR |
| shirt | B-PRODUCT_TYPE | O ⚠️ |
| women | O | O |

---

## Example 12: ❌ ERROR

**Query:** `blue lightsaber`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| blue | B-COLOR | B-COLOR |
| lightsaber | B-PRODUCT_TYPE | O ⚠️ |

---

## Example 13: ❌ ERROR

**Query:** `werewolf toddler costume`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| werewolf | O | O |
| toddler | O | O |
| costume | B-PRODUCT_TYPE | O ⚠️ |

---

## Example 14: ❌ ERROR

**Query:** `martin tshirt women`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| martin | O | O |
| tshirt | O | O |
| women | B-PRODUCT_TYPE | O ⚠️ |

---

## Example 15: ❌ ERROR

**Query:** `2 quart canteen`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| 2 | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| quart | I-PRODUCT_TYPE | I-PRODUCT_TYPE |
| canteen | B-PRODUCT_TYPE | O ⚠️ |

---

## Example 16: ❌ ERROR

**Query:** `open ended toys for 1 year old`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| open | O | O |
| ended | O | O |
| toys | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| for | O | O |
| 1 | O | O |
| year | O | O |
| old | B-PRODUCT_TYPE | O ⚠️ |

---

## Example 17: ❌ ERROR

**Query:** `lion head bracelet for men`

**Error Types:** FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| lion | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| head | I-PRODUCT_TYPE | I-PRODUCT_TYPE |
| bracelet | I-PRODUCT_TYPE | B-PRODUCT_TYPE ⚠️ |
| for | B-PRODUCT_TYPE | O ⚠️ |
| men | I-PRODUCT_TYPE | O ⚠️ |

---

## Example 18: ❌ ERROR

**Query:** `60s wigs for women`

**Error Types:** FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| 60s | O | O |
| wigs | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| for | B-PRODUCT_TYPE | O ⚠️ |
| women | I-PRODUCT_TYPE | B-PRODUCT_TYPE ⚠️ |

---

## Example 19: ❌ ERROR

**Query:** `solid color iphone 8 plus case`

**Error Types:** FP, FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| solid | O | O |
| color | O | B-PRODUCT_TYPE ⚠️ |
| iphone | O | O |
| 8 | B-PRODUCT_TYPE | O ⚠️ |
| plus | I-PRODUCT_TYPE | B-PRODUCT_TYPE ⚠️ |
| case | O | O |

---

## Example 20: ❌ ERROR

**Query:** `white shirts for men`

**Error Types:** TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| white | B-COLOR | B-PRODUCT_TYPE ⚠️ |
| shirts | O | O |
| for | O | O |
| men | O | O |

---

## Example 21: ❌ ERROR

**Query:** `cooking utensils`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| cooking | O | O |
| utensils | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 22: ❌ ERROR

**Query:** `small ivory lamp`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| small | O | B-SIZE_MEASURE ⚠️ |
| ivory | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| lamp | O | O |

---

## Example 23: ❌ ERROR

**Query:** `tripod iphone gopro`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| tripod | O | O |
| iphone | B-PRODUCT_TYPE | O ⚠️ |
| gopro | I-PRODUCT_TYPE | O ⚠️ |

---

## Example 24: ❌ ERROR

**Query:** `side post battery for a`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| side | B-COLOR | O ⚠️ |
| post | I-COLOR | O ⚠️ |
| battery | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| for | O | O |
| a | O | O |

---

## Example 25: ✅ CORRECT

**Query:** `vitamin b 12`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| vitamin | O | O |
| b | O | O |
| 12 | O | O |

---

## Example 26: ✅ CORRECT

**Query:** `holiday gifts for coworkers`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| holiday | O | O |
| gifts | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| for | O | O |
| coworkers | O | O |

---

## Example 27: ✅ CORRECT

**Query:** `backpack for women`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| backpack | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| for | O | O |
| women | O | O |

---

## Example 28: ✅ CORRECT

**Query:** `dragon ball z dvd season 1`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| dragon | O | O |
| ball | O | O |
| z | O | O |
| dvd | O | O |
| season | O | O |
| 1 | O | O |

---

## Example 29: ✅ CORRECT

**Query:** `fake sweets`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| fake | O | O |
| sweets | O | O |

---

## Example 30: ✅ CORRECT

**Query:** `women s skirts below knee`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| women | O | O |
| s | O | O |
| skirts | O | O |
| below | O | O |
| knee | O | O |

---
