# Qualitative Error Analysis: bert_ner

**Total Examples:** 30

---

## Example 1: ❌ ERROR

**Query:** `milwaukee magnetic drill press`

**Error Types:** FP, FN

| Token | Gold | Predicted |
|-------|------|-----------|
| milwaukee | B-BRAND | B-BRAND |
| magnetic | B-ATTRIBUTE_VALUE | B-ATTRIBUTE_VALUE |
| drill | B-PRODUCT_TYPE | O ⚠️ |
| press | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 2: ❌ ERROR

**Query:** `a mid-range spec laptop with a nice screen but lightweight and easy for travelling with.`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| a | O | O |
| mid-range | O | O |
| spec | O | O |
| laptop | O | O |
| with | O | O |
| a | O | O |
| nice | O | O |
| screen | O | O |
| but | O | O |
| lightweight | O | B-ATTRIBUTE_VALUE ⚠️ |
| and | O | O |
| easy | O | O |
| for | O | O |
| travelling | O | O |
| with. | O | O |

---

## Example 3: ❌ ERROR

**Query:** `pink lip gloss`

**Error Types:** FP, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| pink | B-COLOR | B-PRODUCT_TYPE ⚠️ |
| lip | O | O |
| gloss | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 4: ❌ ERROR

**Query:** `chain repair tool`

**Error Types:** FP, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| chain | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| repair | O | B-PRODUCT_TYPE ⚠️ |
| tool | B-PRODUCT_TYPE | I-PRODUCT_TYPE ⚠️ |

---

## Example 5: ❌ ERROR

**Query:** `stoki high chair`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| stoki | O | B-BRAND ⚠️ |
| high | O | O |
| chair | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 6: ❌ ERROR

**Query:** `logger boots`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| logger | O | B-BRAND ⚠️ |
| boots | O | O |

---

## Example 7: ❌ ERROR

**Query:** `poori press`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| poori | O | O |
| press | O | B-PRODUCT_TYPE ⚠️ |

---

## Example 8: ❌ ERROR

**Query:** `plus size sequin dresses for women`

**Error Types:** FP

| Token | Gold | Predicted |
|-------|------|-----------|
| plus | O | B-PRODUCT_TYPE ⚠️ |
| size | O | O |
| sequin | O | O |
| dresses | O | O |
| for | O | O |
| women | O | O |

---

## Example 9: ❌ ERROR

**Query:** `faux copper backsplash`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| faux | O | O |
| copper | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| backsplash | B-PRODUCT_TYPE | O ⚠️ |

---

## Example 10: ❌ ERROR

**Query:** `copper mule mugs set of 2`

**Error Types:** FP, FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| copper | B-MATERIAL | B-MATERIAL |
| mule | B-PRODUCT_TYPE | O ⚠️ |
| mugs | O | B-PRODUCT_TYPE ⚠️ |
| set | O | B-PRODUCT_TYPE ⚠️ |
| of | B-PRODUCT_TYPE | O ⚠️ |
| 2 | I-PRODUCT_TYPE | B-PRODUCT_TYPE ⚠️ |

---

## Example 11: ❌ ERROR

**Query:** `fabric booty bands`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| fabric | O | O |
| booty | O | O |
| bands | B-PRODUCT_TYPE | O ⚠️ |

---

## Example 12: ❌ ERROR

**Query:** `unique soap despenser`

**Error Types:** FP, FN

| Token | Gold | Predicted |
|-------|------|-----------|
| unique | O | O |
| soap | O | B-PRODUCT_TYPE ⚠️ |
| despenser | B-PRODUCT_TYPE | O ⚠️ |

---

## Example 13: ❌ ERROR

**Query:** `bicycle turn signals front and rear`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| bicycle | B-PRODUCT_TYPE | O ⚠️ |
| turn | O | O |
| signals | O | O |
| front | O | O |
| and | O | O |
| rear | O | O |

---

## Example 14: ❌ ERROR

**Query:** `aluminum pie plates`

**Error Types:** FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| aluminum | B-PRODUCT_TYPE | B-MATERIAL ⚠️ |
| pie | I-PRODUCT_TYPE | O ⚠️ |
| plates | I-PRODUCT_TYPE | B-PRODUCT_TYPE ⚠️ |

---

## Example 15: ❌ ERROR

**Query:** `baby bath tub`

**Error Types:** FN

| Token | Gold | Predicted |
|-------|------|-----------|
| baby | O | O |
| bath | O | O |
| tub | B-PRODUCT_TYPE | O ⚠️ |

---

## Example 16: ❌ ERROR

**Query:** `piece bluetooth adapter windows 10`

**Error Types:** FP, FN

| Token | Gold | Predicted |
|-------|------|-----------|
| piece | O | O |
| bluetooth | B-ATTRIBUTE_VALUE | B-ATTRIBUTE_VALUE |
| adapter | O | B-PRODUCT_TYPE ⚠️ |
| windows | B-PRODUCT_TYPE | O ⚠️ |
| 10 | I-PRODUCT_TYPE | O ⚠️ |

---

## Example 17: ❌ ERROR

**Query:** `drafting table`

**Error Types:** FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| drafting | B-PRODUCT_TYPE | O ⚠️ |
| table | I-PRODUCT_TYPE | B-PRODUCT_TYPE ⚠️ |

---

## Example 18: ❌ ERROR

**Query:** `dog toys for small dogs tractors`

**Error Types:** FP, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| dog | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| toys | I-PRODUCT_TYPE | B-PRODUCT_TYPE ⚠️ |
| for | O | O |
| small | O | B-SIZE_MEASURE ⚠️ |
| dogs | O | B-PRODUCT_TYPE ⚠️ |
| tractors | O | O |

---

## Example 19: ❌ ERROR

**Query:** `toker poler`

**Error Types:** TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| toker | B-BRAND | B-BRAND |
| poler | I-BRAND | B-PRODUCT_TYPE ⚠️ |

---

## Example 20: ❌ ERROR

**Query:** `metal stamping blanks copper`

**Error Types:** FP, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| metal | B-MATERIAL | B-MATERIAL |
| stamping | O | B-PRODUCT_TYPE ⚠️ |
| blanks | B-PRODUCT_TYPE | B-PRODUCT_TYPE |
| copper | B-PRODUCT_TYPE | B-MATERIAL ⚠️ |

---

## Example 21: ❌ ERROR

**Query:** `jl amplifier`

**Error Types:** TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| jl | B-PRODUCT_TYPE | B-BRAND ⚠️ |
| amplifier | I-PRODUCT_TYPE | B-PRODUCT_TYPE ⚠️ |

---

## Example 22: ❌ ERROR

**Query:** `catch flights not feelings`

**Error Types:** FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| catch | B-BRAND | O ⚠️ |
| flights | I-BRAND | O ⚠️ |
| not | I-BRAND | O ⚠️ |
| feelings | I-BRAND | I-PRODUCT_TYPE ⚠️ |

---

## Example 23: ❌ ERROR

**Query:** `texh 21 iphone xs max`

**Error Types:** FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| texh | O | O |
| 21 | O | O |
| iphone | B-PRODUCT_TYPE | O ⚠️ |
| xs | I-PRODUCT_TYPE | B-SIZE_MEASURE ⚠️ |
| max | I-PRODUCT_TYPE | O ⚠️ |

---

## Example 24: ❌ ERROR

**Query:** `the girl who spun gold`

**Error Types:** FN, TYPE_CONFUSION

| Token | Gold | Predicted |
|-------|------|-----------|
| the | O | O |
| girl | O | O |
| who | B-PRODUCT_TYPE | O ⚠️ |
| spun | I-PRODUCT_TYPE | O ⚠️ |
| gold | I-PRODUCT_TYPE | B-PRODUCT_TYPE ⚠️ |

---

## Example 25: ✅ CORRECT

**Query:** `office desk fridge`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| office | O | O |
| desk | O | O |
| fridge | O | O |

---

## Example 26: ✅ CORRECT

**Query:** `ladies purses`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| ladies | O | O |
| purses | O | O |

---

## Example 27: ✅ CORRECT

**Query:** `victor mouse trap`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| victor | B-BRAND | B-BRAND |
| mouse | O | O |
| trap | B-PRODUCT_TYPE | B-PRODUCT_TYPE |

---

## Example 28: ✅ CORRECT

**Query:** `lsu shirt men`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| lsu | O | O |
| shirt | O | O |
| men | O | O |

---

## Example 29: ✅ CORRECT

**Query:** `danielle girard dead center`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| danielle | O | O |
| girard | O | O |
| dead | O | O |
| center | O | O |

---

## Example 30: ✅ CORRECT

**Query:** `adidas womens leggings`

**Error Types:** N/A

| Token | Gold | Predicted |
|-------|------|-----------|
| adidas | B-BRAND | B-BRAND |
| womens | O | O |
| leggings | O | O |

---
