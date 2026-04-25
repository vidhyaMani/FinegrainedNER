# Qualitative Error Analysis: roberta_ner

**Total Examples:** 20

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

**Query:** `iowa cyclones womens apparel`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| iowa | O | B-COLOR ⚠️ |
| cyclones | O | O |
| womens | O | O |
| apparel | O | O |

---

## Example 3: ❌ ERROR

**Query:** `expanding file folders`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| expanding | O | O |
| file | O | O |
| folders | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 4: ❌ ERROR

**Query:** `instant ramen beef`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| instant | O | O |
| ramen | O | O |
| beef | O | B-COLOR ⚠️ |

---

## Example 5: ❌ ERROR

**Query:** `narrow end table for livungroom`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| narrow | O | O |
| end | O | O |
| table | O | B-PRODUCT_TYPE ⚠️ |
| for | O | O |
| livungroom | O | O |

---

## Example 6: ❌ ERROR

**Query:** `xbox 360 controller wired white`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| xbox | O | O |
| 360 | O | B-PRODUCT_TYPE ⚠️ |
| controller | O | O |
| wired | O | O |
| white | B-PRODUCT_TYPE | B-PRODUCT_TYPE |

---

## Example 7: ❌ ERROR

**Query:** `halloween costumes adult`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| halloween | O | O |
| costumes | O | O |
| adult | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 8: ❌ ERROR

**Query:** `womens beanie hats`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| womens | O | O |
| beanie | O | O |
| hats | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 9: ❌ ERROR

**Query:** `yellowstone season 1 and 2 dvd set`

**Error Types:** FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| yellowstone | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| season | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| 1 | I-PRODUCT_TYPE | I-PRODUCT_TYPE |
| and | O | O |
| 2 | B-PRODUCT_TYPE | I-PRODUCT_TYPE ⚠️ |
| dvd | O | O |
| set | B-PRODUCT_TYPE | O ⚠️ |

---

## Example 10: ❌ ERROR

**Query:** `mens hooded sweatshirts`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| mens | O | O |
| hooded | O | O |
| sweatshirts | B-PRODUCT_TYPE | O ⚠️ |

---

## Example 11: ❌ ERROR

**Query:** `dragon ball z resurrection f`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| dragon | O | O |
| ball | O | O |
| z | O | O |
| resurrection | B-PRODUCT_TYPE | O ⚠️ |
| f | I-PRODUCT_TYPE | O ⚠️ |

---

## Example 12: ❌ ERROR

**Query:** `police shirt women`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| police | B-COLOR | B-COLOR |
| shirt | B-PRODUCT_TYPE | O ⚠️ |
| women | O | O |

---

## Example 13: ❌ ERROR

**Query:** `white shirts for men`

**Error Types:** TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| white | B-COLOR | B-PRODUCT_TYPE ⚠️ |
| shirts | O | O |
| for | O | O |
| men | O | O |

---

## Example 14: ❌ ERROR

**Query:** `yellowstone season 1 and 2 dvd set`

**Error Types:** FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| yellowstone | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| season | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| 1 | I-PRODUCT_TYPE | I-PRODUCT_TYPE |
| and | O | O |
| 2 | B-PRODUCT_TYPE | I-PRODUCT_TYPE ⚠️ |
| dvd | O | O |
| set | B-PRODUCT_TYPE | O ⚠️ |

---

## Example 15: ✅ CORRECT

**Query:** `garth brooks shirt`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| garth | O | O |
| brooks | O | O |
| shirt | O | O |

---

## Example 16: ✅ CORRECT

**Query:** `light jackets for women`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| light | O | O |
| jackets | O | O |
| for | O | O |
| women | O | O |

---

## Example 17: ✅ CORRECT

**Query:** `lego ninjago costume`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| lego | B-BRAND | B-BRAND |
| ninjago | O | O |
| costume | O | O |

---

## Example 18: ✅ CORRECT

**Query:** `crkt razel`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| crkt | B-BRAND | B-BRAND |
| razel | O | O |

---

## Example 19: ✅ CORRECT

**Query:** `most expensive watch`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| most | O | O |
| expensive | O | O |
| watch | O | O |

---

## Example 20: ✅ CORRECT

**Query:** `iphone x storage flash drive`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| iphone | O | O |
| x | O | O |
| storage | O | O |
| flash | O | O |
| drive | O | O |

---
