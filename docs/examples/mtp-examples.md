# Примеры MTP: где совпало, где разошлось

В корпусе **8409 проверенных пар** из 11 цепочек: **6498** приняты целиком, **1111** — только первый токен, **800** — отказ уже на первом. Это два уже сохранённых набора: calibration100–109 и heldout110; повторного inference нет.

В целиком пробельных токенах `␠` означает один пробел, `↵` — перенос, `⇥` — табуляцию. Успех означает совпадение с greedy target, а не правильность решения задачи. Серый суффикс после первого отказа не является независимо проверенной ошибкой. `p1 × p2` — score драфтера. Контекст и продолжение взяты из реально выданных токенов; предсказания verifier после ошибочного proposal не используются как продолжение.

[Интерактивный каталог всех примеров](mtp-examples.html). Откройте локальный HTML в браузере; GitHub показывает исходный файл. В каталоге есть поиск, фильтры, probabilities, token IDs и ссылки на отдельные раунды.

## Пробелы, отступы и переносы строк

Даже разное число пробелов или один дополнительный перенос дают несовпадение токенов. Это само по себе не означает ошибку в математике.

### c100-r493 · принято 1/2 · calibration

Контекст: `  would have increased (80% -> >80%).↵        *   So the logic holds.↵↵    *   Let's double check the algebra. `

Драфт: ✅ ` ↵ ` · ❌ ` ␠␠␠ `

Реально выдано: ✅ ` ↵ ` · 🔵 ` ␠␠␠␠␠␠␠ `

Продолжение: ` ↵        $4(400-x) = 3(50 `

p1=0.9996; p2=0.9024; score=0.9021. Позиция: 1352 уже выданных токенов. [Открыть карточку](mtp-examples.html#c100-r493).

Перенос строки принят, а токен из трёх пробелов заменён токеном из семи пробелов перед формулой.

### c103-r568 · принято 0/2 · calibration

Контекст: `  Let's stick to the standard interpretation: Base = 24 (the unequal side).↵        *   Wait, could the triangle be obtuse?↵ `

Драфт: ❌ ` ␠␠␠␠␠␠␠␠␠␠␠ ` · ◻️ `  * `

Реально выдано: 🔵 ` ␠␠␠␠␠␠␠ `

Продолжение: `         *   If the base is 24 and height is 5. `

p1=0.9407; p2=0.9952; score=0.9361. Позиция: 1513 уже выданных токенов. [Открыть карточку](mtp-examples.html#c103-r568).

Первый токен содержит 11 пробелов; целевая модель выбирает 7. Различие видно при отображении пробельных символов.

### c106-r219 · принято 1/2 · calibration

Контекст: ` $.*↵        $3N = 76 - 4M$↵        $N = \frac{76 - 4M}{3 `

Драфт: ✅ ` }$ ` · ❌ ` ↵ `

Реально выдано: ✅ ` }$ ` · 🔵 ` ↵↵ `

Продолжение: ` }$↵↵    *   *Step 2: Apply Constraint 2 ($M `

p1=0.9706; p2=0.9313; score=0.9039. Позиция: 606 уже выданных токенов. [Открыть карточку](mtp-examples.html#c106-r219).

Закрытие формулы принято; один перенос строки заменён двумя.

### c101-r128 · принято 2/2 · calibration

Контекст: `  $111^2 = 12321$. Sum of digits = $1+2+3+2+1 = 9$. `

Драфт: ✅ ` ↵ ` · ✅ ` ␠␠␠ `

Реально выдано: ✅ ` ↵ ` · ✅ ` ␠␠␠ ` · 🔵 `  * `

Продолжение: ` ↵    *   Case $k=4$: $1111^ `

p1=0.9993; p2=0.9964; score=0.9957. Позиция: 350 уже выданных токенов. [Открыть карточку](mtp-examples.html#c101-r128).

Пара из переноса строки и отступа полностью принята в повторяющемся списке примеров.

### c105-r148 · принято 0/2 · calibration

Контекст: `  $P < T < Q < S$.↵    *   Combining $P < R$, we know $R$ is slower than $P$.↵↵ `

Драфт: ❌ ` 4 ` · ◻️ ` . `

Реально выдано: 🔵 ` ␠␠␠ `

Продолжение: `     So, the strict ordering chain we have established so far is:↵    `

p1=0.9817; p2=1.0000; score=0.9817. Позиция: 400 уже выданных токенов. [Открыть карточку](mtp-examples.html#c105-r148).

При score=0,982 черновик начинает новый пункт токеном 4, а целевая модель выбирает отступ. Второй токен точки не проверяет независимую альтернативу после исправления.

### c100-r494 · принято 0/2 · calibration

Контекст: `  increased (80% -> >80%).↵        *   So the logic holds.↵↵    *   Let's double check the algebra.↵        `

Драфт: ❌ `  * ` · ◻️ ` ␠␠ `

Реально выдано: 🔵 `  $ `

Продолжение: `  $4(400-x) = 3(500-x `

p1=0.8520; p2=0.9998; score=0.8518. Позиция: 1354 уже выданных токенов. [Открыть карточку](mtp-examples.html#c100-r494).

Сразу после r493 черновик предлагает маркер списка « *», а фактическое продолжение начинается токеном « $», открывающим формулу.

## Пунктуация и границы формул

Совпадение содержательного токена не гарантирует совпадения следующего знака, границы Markdown или команды LaTeX.

### c110-r570 · принято 1/2 · heldout

Контекст: ` ↵    *   $150 - 15x \le 30$.↵    *   $120 \le 15 `

Драфт: ✅ ` x ` · ❌ `  \ `

Реально выдано: ✅ ` x ` · 🔵 ` $. `

Продолжение: ` x$.↵    *   $x \ge 8$.↵↵    * `

p1=0.9943; p2=0.9794; score=0.9738. Позиция: 1478 уже выданных токенов. [Открыть карточку](mtp-examples.html#c110-r570).

x принят, но пробел с обратной косой чертой заменён токеном «$.». Score=0,974: p1≈0,994 и p2≈0,979 не исключили отказ на втором токене.

### c107-r405 · принято 1/2 · calibration

Контекст: `  else? No, "percent of the rectangle's area is inside the square" means $\frac{\text{Area of intersection}}{\text{Area of rectangle `

Драфт: ✅ ` }} ` · ❌ ` $. `

Реально выдано: ✅ ` }} ` · 🔵 `  \ `

Продолжение: ` }} \times 100$. Since the square is inside the rectangle, `

p1=0.9980; p2=0.9234; score=0.9216. Позиция: 1063 уже выданных токенов. [Открыть карточку](mtp-examples.html#c107-r405).

Закрывающие фигурные скобки приняты; вместо немедленного завершения формулы целевая модель продолжает умножением на 100.

### c101-r351 · принято 1/2 · calibration

Контекст: ` 4+5+6+7+8+9)$.↵↵7.  **Perform the Calculation:**↵    *   Sum of integers from 1 `

Драфт: ✅ `  to ` · ❌ `  $ `

Реально выдано: ✅ `  to ` · 🔵 ` ␠ `

Продолжение: `  to 9:↵        $S_9 = \frac{9 \ `

p1=0.9999; p2=0.9307; score=0.9307. Позиция: 960 уже выданных токенов. [Открыть карточку](mtp-examples.html#c101-r351).

« to» принято; перед 9 целевая модель выбирает обычный пробел вместо начала математического фрагмента « $».

### c104-r26 · принято 2/2 · calibration

Контекст: `  the decimal point needed to express a specific fraction as a decimal.↵↵2.  **Identify the Fraction:**↵    $$ \text{Fraction} = `

Драфт: ✅ `  \ ` · ✅ ` frac `

Реально выдано: ✅ `  \ ` · ✅ ` frac ` · 🔵 ` { `

Продолжение: `  \frac{123456789}{2^{2 `

p1=0.9993; p2=0.9978; score=0.9971. Позиция: 67 уже выданных токенов. [Открыть карточку](mtp-examples.html#c104-r26).

Оба токена начала команды LaTeX для дроби приняты.

### c101-r46 · принято 1/2 · calibration

Контекст: ` ↵↵2.  **Identify the Number:**↵    *   The number $n$ consists of nine $1$s.↵    *   $n `

Драфт: ✅ `  = ` · ❌ ` ␠ `

Реально выдано: ✅ `  = ` · 🔵 `  \ `

Продолжение: `  = \underbrace{111,111,111 `

p1=0.9803; p2=0.9653; score=0.9463. Позиция: 130 уже выданных токенов. [Открыть карточку](mtp-examples.html#c101-r46).

« =» принято; обычный пробел заменён пробелом с обратной косой чертой перед командой underbrace. Score=0,946.

### c101-r203 · принято 1/2 · calibration

Контекст: ` 's verify this pattern holds for $k=9$.↵    *   The number is $R_9$.↵    *   $R_9^ `

Драфт: ✅ ` 2 ` · ❌ `  = `

Реально выдано: ✅ ` 2 ` · 🔵 ` $ `

Продолжение: ` 2$ should look like $1234567898 `

p1=1.0000; p2=0.9015; score=0.9015. Позиция: 553 уже выданных токенов. [Открыть карточку](mtp-examples.html#c101-r203).

Цифра 2 принята при p1≈1; « =» отклонено при p2≈0,902. Фактическое продолжение закрывает формулу знаком $ и переходит к словам.

## Выбор слов и стиля

Показаны разные продолжения одной фразы. По одному отказу нельзя заключить, что отвергнутая формулировка неверна по смыслу.

### c104-r618 · принято 1/2 · calibration

Контекст: `  We need to multiply the numerator and denominator by $2^a \cdot 5^b$ such that the exponents of 2 and 5 in `

Драфт: ✅ `  the ` · ❌ `  denominator `

Реально выдано: ✅ `  the ` · 🔵 `  new `

Продолжение: `  the new denominator are equal.↵    New denominator $D' = 2 `

p1=0.9847; p2=0.9729; score=0.9580. Позиция: 1626 уже выданных токенов. [Открыть карточку](mtp-examples.html#c104-r618).

« the» принято; « denominator» заменено на « new»: фактическая фраза — «the new denominator». Score=0,958.

### c107-r313 · принято 0/2 · calibration

Контекст: `  inside the square?"↵    *   This phrasing is slightly ambiguous.↵        *   Interpretation A: The square is just placed somewhere inside. If `

Драфт: ❌ `  so ` · ◻️ ` , `

Реально выдано: 🔵 `  it `

Продолжение: `  it's just "a" square, the area inside the square is just the `

p1=0.9421; p2=0.9931; score=0.9356. Позиция: 822 уже выданных токенов. [Открыть карточку](mtp-examples.html#c107-r313).

Первое « so» отклонено в пользу « it» внутри пояснения. Запятую во втором токене нельзя считать отдельной ошибкой на исправленном префиксе.

### c110-r138 · принято 1/2 · heldout

Контекст: `  miles per hour} = 0.25 \text{ hours}$.↵↵5.  **Convert Units:**↵    *   The rowing speed `

Драфт: ✅ `  is ` · ❌ `  given `

Реально выдано: ✅ `  is ` · 🔵 `  in `

Продолжение: `  is in miles per hour.↵    *   The water rates are in gallons `

p1=0.9623; p2=0.9121; score=0.8777. Позиция: 363 уже выданных токенов. [Открыть карточку](mtp-examples.html#c110-r138).

« is» принято; « given» заменено на « in» в описании единиц измерения.

### c100-r477 · принято 2/2 · calibration

Контекст: `  Red balls.↵        *   80% -> 75% is a decrease. This direction makes sense.↵        *   If we had `

Драфт: ✅ `  removed ` · ✅ `  Blue `

Реально выдано: ✅ `  removed ` · ✅ `  Blue ` · 🔵 `  balls `

Продолжение: `  removed Blue balls, the percentage of Red balls would have increased (80% `

p1=0.3621; p2=0.2607; score=0.0944. Позиция: 1311 уже выданных токенов. [Открыть карточку](mtp-examples.html#c100-r477).

Оба токена « removed Blue» приняты при низком score≈0,094.

### c104-r124 · принято 1/2 · calibration

Контекст: ` x \cdot 5^y}$ can be written as a terminating decimal.↵    To find the number of decimal places, we need to make the ex `

Драфт: ✅ ` ponents ` · ❌ `  equal `

Реально выдано: ✅ ` ponents ` · 🔵 `  of `

Продолжение: ` ponents of 2 and 5 equal.↵    Let the fraction be $\ `

p1=1.0000; p2=0.8842; score=0.8842. Позиция: 325 уже выданных токенов. [Открыть карточку](mtp-examples.html#c104-r124).

После ex окончание ponents принято при p1≈1; « equal» заменено на « of». Фактическое продолжение: «exponents of 2 and 5 equal».

### c110-r276 · принято 1/2 · heldout

Контекст: `    $x \ge 120 / 15$↵    *   $x \ge 8$↵↵    *   So, Le `

Драфт: ✅ ` Roy ` · ❌ `  must `

Реально выдано: ✅ ` Roy ` · 🔵 `  needs `

Продолжение: ` Roy needs to bail at least 8 gallons per minute.↵↵8.  `

p1=0.9898; p2=0.3632; score=0.3594. Позиция: 738 уже выданных токенов. [Открыть карточку](mtp-examples.html#c110-r276).

После Le продолжение имени Roy принято, а « must» заменено на « needs». Это различие в продолжении фразы, а не отказ на имени.

## Цифры и числовые последовательности

В этих примерах расходятся конкретные цифры, а не только оформление числа. Рядом встречаются длинные последовательности точных совпадений. Это не оценка общей математической правильности.

### c104-r439 · принято 1/2 · calibration

Контекст: `  question asks for the number of digits needed to *express* the fraction.↵    If the fraction is $\frac{1}{20} = 0 `

Драфт: ✅ ` . ` · ❌ ` 5 `

Реально выдано: ✅ ` . ` · 🔵 ` 0 `

Продолжение: ` .05$, we need 2 digits.↵    If the fraction is `

p1=0.9990; p2=0.9091; score=0.9082. Позиция: 1143 уже выданных токенов. [Открыть карточку](mtp-examples.html#c104-r439).

После 1/20 = 0 точка принята, но 5 заменено на 0. Фактическое продолжение — .05. Score=0,908.

### c101-r546 · принято 1/2 · calibration

Контекст: `  coefficients for $k=9$.↵        *   $i+j=9$. Pairs: $(1,8), (2,7), (3 `

Драфт: ✅ ` , ` · ❌ ` 8 `

Реально выдано: ✅ ` , ` · 🔵 ` 6 `

Продолжение: ` ,6), (4,5), (5,4), (6, `

p1=0.9996; p2=0.8417; score=0.8414. Позиция: 1462 уже выданных токенов. [Открыть карточку](mtp-examples.html#c101-r546).

Запятая принята; 8 заменено на 6 в паре (3,6). Score=0,841.

### c101-r135 · принято 2/2 · calibration

Контекст: ` 2+3+2+1 = 9$.↵    *   Case $k=4$: $1111^2 = 12 `

Драфт: ✅ ` 3 ` · ✅ ` 4 `

Реально выдано: ✅ ` 3 ` · ✅ ` 4 ` · 🔵 ` 3 `

Продолжение: ` 34321$. Sum of digits = $1+2+3 `

p1=0.9980; p2=0.7301; score=0.7286. Позиция: 371 уже выданных токенов. [Открыть карточку](mtp-examples.html#c101-r135).

Обе цифры 3 и 4 приняты внутри записи 1234321.

### c110-r561 · принято 2/2 · heldout

Контекст: `  30$.↵    *   $10(15) - x(15) \le 30$.↵    *   $ `

Драфт: ✅ ` 1 ` · ✅ ` 5 `

Реально выдано: ✅ ` 1 ` · ✅ ` 5 ` · 🔵 ` 0 `

Продолжение: ` 150 - 15x \le 30$.↵    `

p1=0.9999; p2=0.9999; score=0.9998. Позиция: 1451 уже выданных токенов. [Открыть карточку](mtp-examples.html#c110-r561).

Обе цифры 1 и 5 приняты в начале числа 150 в балансе воды.

### c101-r120 · принято 1/2 · calibration

Контекст: `  of digits = $1+2+1 = 4$.↵    *   Case $k=3$: $111^2 = 1 `

Драфт: ✅ ` 2 ` · ❌ ` 1 `

Реально выдано: ✅ ` 2 ` · 🔵 ` 3 `

Продолжение: ` 2321$. Sum of digits = $1+2+3+ `

p1=0.9991; p2=0.7792; score=0.7785. Позиция: 327 уже выданных токенов. [Открыть карточку](mtp-examples.html#c101-r120).

В записи 111² после уже выведенной 1 черновик предлагает 2,1: 2 принята, 1 заменена на 3. Следующий r121 уже полностью принят.

### c101-r141 · принято 2/2 · calibration

Контекст: ` =4$: $1111^2 = 1234321$. Sum of digits = $1+2+3+4 `

Драфт: ✅ ` + ` · ✅ ` 3 `

Реально выдано: ✅ ` + ` · ✅ ` 3 ` · 🔵 ` + `

Продолжение: ` +3+2+1 = 16$.↵    *   Case `

p1=0.9994; p2=0.4943; score=0.4940. Позиция: 389 уже выданных токенов. [Открыть карточку](mtp-examples.html#c101-r141).

Плюс и цифра 3 полностью приняты, хотя score≈0,494; низкая уверенность второго токена не помешала совпадению.

## Алгебра и математическая запись

Одно несовпадение может менять запись выражения. Соседние раунды показывают, как после исправления вновь принимаются оба токена. Причина выбора черновика из этих записей неизвестна.

### c109-r257 · принято 0/2 · calibration

Контекст: ` }{a}\right)$↵        $(\alpha - \beta)^2 = \frac{b^2}{a^2} - \frac{4 `

Драфт: ❌ ` ac ` · ◻️ ` }{ `

Реально выдано: 🔵 ` c `

Продолжение: ` c}{a}$↵        $(\alpha - \beta)^2 = \ `

p1=0.9942; p2=0.9967; score=0.9909. Позиция: 735 уже выданных токенов. [Открыть карточку](mtp-examples.html#c109-r257).

После минуса и начала дроби с числителем 4 токен ac отклонён в пользу c; далее фактически идёт c}{a}. Score≈0,991 — максимум среди отказов в этих данных.

### c104-r232 · принято 0/2 · calibration

Контекст: ` 0, y=1$. $\max(0,1)=1$.↵    $\frac{1}{20} = \frac{1}{2 `

Драфт: ❌ `  \ ` · ◻️ ` cdot `

Реально выдано: 🔵 ` ^ `

Продолжение: ` ^2 \cdot 5^1} = \frac{5}{2 `

p1=0.9132; p2=0.9707; score=0.8864. Позиция: 612 уже выданных токенов. [Открыть карточку](mtp-examples.html#c104-r232).

После 2 в знаменателе примера 1/20 пробел с обратной косой чертой отклонён в пользу ^: целевая модель записывает степень.

### c109-r24 · принято 2/2 · calibration

Контекст: `  $x^2 - 7x - 9 = 0$.↵↵2.  **Identify the Equation:**↵    $ax^2 + `

Драфт: ✅ `  bx ` · ✅ `  + `

Реально выдано: ✅ `  bx ` · ✅ `  + ` · 🔵 `  c `

Продолжение: `  bx + c = 0$↵    Here, $a = 1 `

p1=0.9976; p2=0.9800; score=0.9777. Позиция: 66 уже выданных токенов. [Открыть карточку](mtp-examples.html#c109-r24).

Оба токена « bx +» приняты в общем виде квадратного уравнения.

### c108-r78 · принято 2/2 · calibration

Контекст: ` $ to exist, the sum of any two sides must be greater than the third side.↵    *   Conditions:↵        1.  $a `

Драфт: ✅ `  + ` · ✅ `  b `

Реально выдано: ✅ `  + ` · ✅ `  b ` · 🔵 `  > `

Продолжение: `  + b > c$↵        2.  $a + c > `

p1=0.9958; p2=0.8782; score=0.8746. Позиция: 215 уже выданных токенов. [Открыть карточку](mtp-examples.html#c108-r78).

Оба токена « + b» приняты в неравенстве треугольника.

### c109-r258 · принято 2/2 · calibration

Контекст: ` a}\right)$↵        $(\alpha - \beta)^2 = \frac{b^2}{a^2} - \frac{4c `

Драфт: ✅ ` }{ ` · ✅ ` a `

Реально выдано: ✅ ` }{ ` · ✅ ` a ` · 🔵 ` }$ `

Продолжение: ` }{a}$↵        $(\alpha - \beta)^2 = \frac `

p1=0.9985; p2=0.9314; score=0.9300. Позиция: 736 уже выданных токенов. [Открыть карточку](mtp-examples.html#c109-r258).

Сразу после исправления в r257 пара «}{» и «a» полностью принята. Это уже новый раунд с исправленным префиксом.

### c104-r233 · принято 2/2 · calibration

Контекст: ` , y=1$. $\max(0,1)=1$.↵    $\frac{1}{20} = \frac{1}{2^ `

Драфт: ✅ ` 2 ` · ✅ `  \ `

Реально выдано: ✅ ` 2 ` · ✅ `  \ ` · 🔵 ` cdot `

Продолжение: ` 2 \cdot 5^1} = \frac{5}{2^ `

p1=0.9988; p2=0.9103; score=0.9092. Позиция: 613 уже выданных токенов. [Открыть карточку](mtp-examples.html#c104-r233).

После вставленного в r232 знака ^ оба предложения — 2 и пробел с обратной косой чертой — приняты; далее идёт команда cdot.

## Имена и обозначения переменных

Имена и буквенные метки тоже состоят из токенов. Несовпадение около имени не доказывает, что модель перепутала человека.

### c105-r421 · принято 1/2 · calibration

Контекст: `  R, S\}$ into positions $\{2, 3, 4, 5\}$ such that $T$ comes before $Q$ and `

Драфт: ✅ `  $ ` · ❌ ` S `

Реально выдано: ✅ `  $ ` · 🔵 ` Q `

Продолжение: `  $Q$ comes before $S$.↵↵    Let's analyze the possible positions `

p1=0.9976; p2=0.9522; score=0.9499. Позиция: 1101 уже выданных токенов. [Открыть карточку](mtp-examples.html#c105-r421).

Начало математического фрагмента принято; S заменено на Q в цепочке ограничений для бегунов. Score=0,950.

### c110-r243 · принято 1/2 · heldout

Контекст: `  So, $150 - (x \times 15) \le 30$.↵↵7.  **Solve for $x$ `

Драфт: ✅ `  ( ` · ❌ ` Rate `

Реально выдано: ✅ `  ( ` · 🔵 ` Le `

Продолжение: `  (LeRoy's rate):**↵    *   $150 -  `

p1=0.9772; p2=0.7757; score=0.7580. Позиция: 644 уже выданных токенов. [Открыть карточку](mtp-examples.html#c110-r243).

Открывающая скобка принята; Rate заменено на Le. Фактическое продолжение — «LeRoy's rate».

### c110-r55 · принято 2/2 · heldout

Контекст: ` }$): 30 gallons.↵    *   Steve's rowing speed ($v_{Steve}$): 4 miles per hour.↵    *   `

Драфт: ✅ `  Le ` · ✅ ` Roy `

Реально выдано: ✅ `  Le ` · ✅ ` Roy ` · 🔵 ` 's `

Продолжение: `  LeRoy's bailing rate ($R_{out}$): Unknown, let's `

p1=0.9425; p2=0.9839; score=0.9273. Позиция: 153 уже выданных токенов. [Открыть карточку](mtp-examples.html#c110-r55).

Два токена полного имени « LeRoy» приняты.

### c110-r244 · принято 2/2 · heldout

Контекст: `  $150 - (x \times 15) \le 30$.↵↵7.  **Solve for $x$ (Le `

Драфт: ✅ ` Roy ` · ✅ ` 's `

Реально выдано: ✅ ` Roy ` · ✅ ` 's ` · 🔵 `  rate `

Продолжение: ` Roy's rate):**↵    *   $150 - 15 `

p1=0.9947; p2=0.9011; score=0.8963. Позиция: 646 уже выданных токенов. [Открыть карточку](mtp-examples.html#c110-r244).

После фактически выведенного Le оба токена Roy и притяжательного окончания приняты.

### c110-r330 · принято 2/2 · heldout

Контекст: `  there's a trick regarding the *distribution* of work or the *timing*.↵    *   However, the problem states "Steve starts rowing... while `

Драфт: ✅ `  Le ` · ✅ ` Roy `

Реально выдано: ✅ `  Le ` · ✅ ` Roy ` · 🔵 `  b `

Продолжение: `  LeRoy bails". This implies they are doing it simultaneously.↵    * `

p1=0.9994; p2=0.9953; score=0.9946. Позиция: 875 уже выданных токенов. [Открыть карточку](mtp-examples.html#c110-r330).

Повторное имя « LeRoy» целиком принято с высоким score≈0,995.

### c110-r307 · принято 2/2 · heldout

Контекст: `    "Steve starts rowing towards the shore... LeRoy bails water out."↵    *   "What is the slowest rate... at which Le `

Драфт: ✅ ` Roy ` · ✅ `  can `

Реально выдано: ✅ ` Roy ` · ✅ `  can ` · 🔵 `  bail `

Продолжение: ` Roy can bail if they are to reach the shore without sinking?"↵    * `

p1=0.9970; p2=0.9973; score=0.9943. Позиция: 817 уже выданных токенов. [Открыть карточку](mtp-examples.html#c110-r307).

После Le оба токена «Roy can» приняты с score≈0,994.

## Границы внутри слов

Граница токена часто проходит внутри английского слова. Первый фрагмент может совпасть, а следующий токен продолжить фразу иначе.

### c104-r711 · принято 0/2 · calibration

Контекст: ` ^{22}}{10^{26}}$.↵    This represents a decimal with 26 digits.↵    The value is $0.d `

Драфт: ❌ ` igits ` · ◻️ ` (N `

Реально выдано: 🔵 ` _ `

Продолжение: ` _1 d_2 \dots d_{26}$.↵↵    Is `

p1=0.4471; p2=0.1271; score=0.0568. Позиция: 1879 уже выданных токенов. [Открыть карточку](mtp-examples.html#c104-r711).

После 0.d черновое igits отклонено в пользу подчёркивания: целевая модель продолжает d_1, а не слово digits.

### c107-r275 · принято 1/2 · calibration

Контекст: `    *Wait, let's re-read carefully.* "A square is drawn inside a rectangle."↵    *   Does this mean the square is the *larg `

Драфт: ✅ ` est ` · ❌ ` * `

Реально выдано: ✅ ` est ` · 🔵 `  possible `

Продолжение: ` est possible* square inside? Or just *a* square?↵    * `

p1=0.9523; p2=0.9803; score=0.9336. Позиция: 727 уже выданных токенов. [Открыть карточку](mtp-examples.html#c107-r275).

После larg окончание est принято; закрывающая звёздочка заменена на « possible».

### c105-r667 · принято 1/2 · calibration

Контекст: `  be after $P$ (1st).↵    $R$ is not constrained relative to $T, Q, S$ directly, other than the trans `

Драфт: ✅ ` itive ` · ❌ `  property `

Реально выдано: ✅ ` itive ` · 🔵 `  chain `

Продолжение: ` itive chain.↵    Can $R$ be 3rd?↵    `

p1=0.7309; p2=0.8433; score=0.6163. Позиция: 1786 уже выданных токенов. [Открыть карточку](mtp-examples.html#c105-r667).

После trans окончание itive принято; « property» заменено на « chain».

### c110-r484 · принято 1/2 · heldout

Контекст: ` ." This phrasing usually implies a cumulative total. If the total water ever exceeds 30, it sinks.↵    *   If the total water accum `

Драфт: ✅ ` ulates ` · ❌ `  over `

Реально выдано: ✅ ` ulates ` · 🔵 `  linear `

Продолжение: ` ulates linearly (constant rate in, constant rate out), the maximum amount of `

p1=0.9960; p2=0.3876; score=0.3860. Позиция: 1257 уже выданных токенов. [Открыть карточку](mtp-examples.html#c110-r484).

После accum окончание ulates принято; « over» заменено на первый фрагмент слова linearly.

## Низкая уверенность, но точное совпадение

Произведение вероятностей черновика — не гарантия принятия и не откалиброванная вероятность успеха. Даже очень малый score иногда сопровождается принятием обеих позиций.

### c106-r647 · принято 2/2 · calibration

Контекст: `  plays every other team in its division $N$ games." This usually means $N$ games total against that specific opponent.↵    *   If it meant `

Драфт: ✅ `  " ` · ✅ ` plays `

Реально выдано: ✅ `  " ` · ✅ ` plays ` · 🔵 `  a `

Продолжение: `  "plays a schedule of $N$ games against every other team", that's `

p1=0.2199; p2=0.2043; score=0.0449. Позиция: 1817 уже выданных токенов. [Открыть карточку](mtp-examples.html#c106-r647).

Токен пробела с открывающей кавычкой и plays оба приняты. Score≈0,045 — минимум среди полностью принятых пар в этих данных.

### c105-r23 · принято 2/2 · calibration

Контекст: `  are 5 runners: $P, Q, R, S, T$.↵    *   There are specific constraints on their finishing order (let's denote `

Драфт: ✅ `  the ` · ✅ `  position `

Реально выдано: ✅ `  the ` · ✅ `  position ` · 🔵 `  as `

Продолжение: `  the position as $1, 2, 3, 4,  `

p1=0.3189; p2=0.1924; score=0.0613. Позиция: 58 уже выданных токенов. [Открыть карточку](mtp-examples.html#c105-r23).

« the position» полностью принято при score≈0,061.

### c103-r576 · принято 2/2 · calibration

Контекст: `  *   Wait, could the triangle be obtuse?↵        *   If the base is 24 and height is 5.↵        *   `

Драфт: ✅ `  The ` · ✅ `  legs `

Реально выдано: ✅ `  The ` · ✅ `  legs ` · 🔵 `  are `

Продолжение: `  The legs are 13.↵        *   $13+1 `

p1=0.2070; p2=0.3064; score=0.0634. Позиция: 1533 уже выданных токенов. [Открыть карточку](mtp-examples.html#c103-r576).

« The legs» полностью принято при score≈0,063.

### c102-r33 · принято 2/2 · calibration

Контекст: `  data or ask for it. *Wait, looking at the prompt, it's a text-based interaction. Usually, in these scenarios, the user provides the data `

Драфт: ✅ `  or ` · ✅ `  the `

Реально выдано: ✅ `  or ` · ✅ `  the ` · 🔵 `  image `

Продолжение: `  or the image is embedded. Since I am an AI text model, I cannot `

p1=0.2656; p2=0.3858; score=0.1025. Позиция: 80 уже выданных токенов. [Открыть карточку](mtp-examples.html#c102-r33).

« or the» полностью принято при score≈0,102 в рассуждении об отсутствующем графике.

### c109-r320 · принято 2/2 · calibration

Контекст: `  - \beta|$. Sometimes it might imply the algebraic difference $\alpha - \beta$ (which could be negative). However, in multiple-choice contexts or `

Драфт: ✅ `  general ` · ✅ `  math `

Реально выдано: ✅ `  general ` · ✅ `  math ` · 🔵 `  problems `

Продолжение: `  general math problems, "difference" usually refers to the magnitude or the positive difference `

p1=0.3444; p2=0.2265; score=0.0780. Позиция: 912 уже выданных токенов. [Открыть карточку](mtp-examples.html#c109-r320).

« general math» полностью принято при score≈0,078.

### c109-r611 · принято 2/2 · calibration

Контекст: `  the larger root and the smaller root" or similar. I will provide $\sqrt{85}$.↵↵    Let's check if there are any trick questions `

Драфт: ✅ `  or ` · ✅ `  specific `

Реально выдано: ✅ `  or ` · ✅ `  specific ` · 🔵 `  integer `

Продолжение: `  or specific integer constraints. No, it's a general quadratic.↵↵    Final `

p1=0.6723; p2=0.1175; score=0.0790. Позиция: 1690 уже выданных токенов. [Открыть карточку](mtp-examples.html#c109-r611).

« or specific» полностью принято при score≈0,079; вероятность второго выбранного токена всего около 0,118.

### c110-r784 · принято 2/2 · heldout

Контекст: ` Roy can bail water *while* Steve rows, but maybe LeRoy can stop rowing? No, Steve rows.↵    *   Maybe LeRoy can `

Драфт: ✅ `  bail ` · ✅ `  water `

Реально выдано: ✅ `  bail ` · ✅ `  water ` · 🔵 `  * `

Продолжение: `  bail water *faster* than the leak? `

p1=0.3181; p2=0.2941; score=0.0936. Позиция: 2038 уже выданных токенов. [Открыть карточку](mtp-examples.html#c110-r784).

« bail water» полностью принято при score≈0,094.

### c110-r338 · принято 2/2 · heldout

Контекст: `  the problem states "Steve starts rowing... while LeRoy bails". This implies they are doing it simultaneously.↵    *   Is there a constraint on `

Драфт: ✅ `  the ` · ✅ `  boat `

Реально выдано: ✅ `  the ` · ✅ `  boat ` · 🔵 ` 's `

Продолжение: `  the boat's capacity? Yes, 30 gallons.↵    *   `

p1=0.3794; p2=0.2741; score=0.1040. Позиция: 897 уже выданных токенов. [Открыть карточку](mtp-examples.html#c110-r338).

« the boat» полностью принято при score≈0,104.

## Несколько отказов подряд

Каждое исправление меняет префикс следующего раунда. Здесь видна последовательность отдельных проверок на трудном участке, а не повторная проверка одинакового входа.

### c102-r49 · принято 0/2 · calibration

Контекст: `  image is embedded. Since I am an AI text model, I cannot "see" an image unless it's uploaded. However, often these problems come from a `

Драфт: ❌ `  dataset ` · ◻️ `  where `

Реально выдано: 🔵 `  specific `

Продолжение: `  specific context or dataset. Let's look closer at the prompt. It says " `

p1=0.2852; p2=0.8686; score=0.2477. Позиция: 114 уже выданных токенов. [Открыть карточку](mtp-examples.html#c102-r49).

Начало пяти последовательных раундов с нулём принятых токенов: « dataset» отклонено в пользу « specific».

### c102-r50 · принято 0/2 · calibration

Контекст: `  is embedded. Since I am an AI text model, I cannot "see" an image unless it's uploaded. However, often these problems come from a specific `

Драфт: ❌ `  dataset ` · ◻️ `  where `

Реально выдано: 🔵 `  context `

Продолжение: `  context or dataset. Let's look closer at the prompt. It says "The `

p1=0.3539; p2=0.5494; score=0.1945. Позиция: 115 уже выданных токенов. [Открыть карточку](mtp-examples.html#c102-r50).

Следующий раунд снова предлагает « dataset», но теперь фактическое исправление — « context».

### c102-r51 · принято 0/2 · calibration

Контекст: `  embedded. Since I am an AI text model, I cannot "see" an image unless it's uploaded. However, often these problems come from a specific context `

Драфт: ❌ `  where ` · ◻️ `  the `

Реально выдано: 🔵 `  or `

Продолжение: `  or dataset. Let's look closer at the prompt. It says "The following `

p1=0.5036; p2=0.5294; score=0.2666. Позиция: 116 уже выданных токенов. [Открыть карточку](mtp-examples.html#c102-r51).

После «specific context» первый токен « where» заменён на « or»; второй токен черновика уже зависит от отвергнутого первого.

### c102-r52 · принято 0/2 · calibration

Контекст: ` . Since I am an AI text model, I cannot "see" an image unless it's uploaded. However, often these problems come from a specific context or `

Драфт: ❌ `  a ` · ◻️ `  known `

Реально выдано: 🔵 `  dataset `

Продолжение: `  dataset. Let's look closer at the prompt. It says "The following bar `

p1=0.3808; p2=0.3672; score=0.1398. Позиция: 117 уже выданных токенов. [Открыть карточку](mtp-examples.html#c102-r52).

После «specific context or» первый токен « a» заменён на « dataset».

### c102-r53 · принято 0/2 · calibration

Контекст: `  Since I am an AI text model, I cannot "see" an image unless it's uploaded. However, often these problems come from a specific context or dataset `

Драфт: ❌ `  where ` · ◻️ `  the `

Реально выдано: 🔵 ` . `

Продолжение: ` . Let's look closer at the prompt. It says "The following bar graph `

p1=0.7507; p2=0.7714; score=0.5791. Позиция: 118 уже выданных токенов. [Открыть карточку](mtp-examples.html#c102-r53).

Пятый нулевой раунд подряд: после dataset предложение « where» отклонено в пользу точки.

### c105-r666 · принято 0/2 · calibration

Контекст: `  must be after $P$ (1st).↵    $R$ is not constrained relative to $T, Q, S$ directly, other than the `

Драфт: ❌ `  general ` · ◻️ `  position `

Реально выдано: 🔵 `  trans `

Продолжение: `  transitive chain.↵    Can $R$ be 3rd?↵ `

p1=0.1735; p2=0.2236; score=0.0388. Позиция: 1785 уже выданных токенов. [Открыть карточку](mtp-examples.html#c105-r666).

Внутри восьми раундов без полного принятия (661–668) токен « general» заменён на « trans» в будущем слове transitive.

### c110-r486 · принято 0/2 · heldout

Контекст: `  usually implies a cumulative total. If the total water ever exceeds 30, it sinks.↵    *   If the total water accumulates linearly ( `

Драфт: ❌ ` In ` · ◻️ `  = `

Реально выдано: 🔵 ` constant `

Продолжение: ` constant rate in, constant rate out), the maximum amount of water in the boat `

p1=0.2316; p2=0.2311; score=0.0535. Позиция: 1261 уже выданных токенов. [Открыть карточку](mtp-examples.html#c110-r486).

Внутри восьми раундов без полного принятия на heldout (482–489) In заменено на constant. Истинное продолжение: «constant rate in, constant rate out».

## Длинные участки полного принятия

Длинные серии совпадений соседствуют с сериями отказов. Принятие означает согласие с целевыми токенами, а не подтверждение истинности всего рассуждения.

### c101-r121 · принято 2/2 · calibration

Контекст: `  = $1+2+1 = 4$.↵    *   Case $k=3$: $111^2 = 123 `

Драфт: ✅ ` 2 ` · ✅ ` 1 `

Реально выдано: ✅ ` 2 ` · ✅ ` 1 ` · 🔵 ` $. `

Продолжение: ` 21$. Sum of digits = $1+2+3+2+ `

p1=0.9890; p2=0.7995; score=0.7907. Позиция: 329 уже выданных токенов. [Открыть карточку](mtp-examples.html#c101-r121).

Начало 44 полностью принятых раундов подряд (121–164) в повторяющихся примерах квадратов чисел из единиц.

### c101-r164 · принято 2/2 · calibration

Контекст: ` 321$. Sum of digits = $1+2+3+4+5+4+3+2+1 = 25$.↵↵ `

Драфт: ✅ ` 4 ` · ✅ ` . `

Реально выдано: ✅ ` 4 ` · ✅ ` . ` · 🔵 ` ␠ `

Продолжение: ` 4.  **Generalize the Pattern:**↵    *   It seems that `

p1=0.6222; p2=0.9999; score=0.6221. Позиция: 458 уже выданных токенов. [Открыть карточку](mtp-examples.html#c101-r164).

Конец той же серии из 44 раундов: «4.» принято как начало следующего пункта, хотя score≈0,622.

### c102-r736 · принято 2/2 · calibration

Контекст: `  12)↵    *   Median is 5.↵↵    *   *Wait, what if the data is:*↵        *   3 `

Драфт: ✅ ` : ` · ✅ ` ␠ `

Реально выдано: ✅ ` : ` · ✅ ` ␠ ` · 🔵 ` 4 `

Продолжение: ` : 4↵        *   4: 5↵        *   `

p1=0.8412; p2=0.9801; score=0.8245. Позиция: 1903 уже выданных токенов. [Открыть карточку](mtp-examples.html#c102-r736).

Начало 48 полностью принятых раундов подряд (736–783) в гипотетической таблице. Согласие токенов не подтверждает, что эти данные были на отсутствующем графике.

### c110-r569 · принято 2/2 · heldout

Контекст: ` 30$.↵    *   $150 - 15x \le 30$.↵    *   $120 \le `

Драфт: ✅ ` ␠ ` · ✅ ` 1 `

Реально выдано: ✅ ` ␠ ` · ✅ ` 1 ` · 🔵 ` 5 `

Продолжение: `  15x$.↵    *   $x \ge 8$. `

p1=1.0000; p2=1.0000; score=1.0000. Позиция: 1475 уже выданных токенов. [Открыть карточку](mtp-examples.html#c110-r569).

Конец 22 полностью принятых раундов подряд (548–569) в алгебре баланса воды. Следующий r570 отказывает на границе формулы при высоком score.

### c101-r153 · принято 2/2 · calibration

Контекст: ` +1 = 16$.↵    *   Case $k=5$: $11111^2 = 12345 `

Драфт: ✅ ` 4 ` · ✅ ` 3 `

Реально выдано: ✅ ` 4 ` · ✅ ` 3 ` · 🔵 ` 2 `

Продолжение: ` 4321$. Sum of digits = $1+2+3+ `

p1=0.9824; p2=0.5259; score=0.5166. Позиция: 425 уже выданных токенов. [Открыть карточку](mtp-examples.html#c101-r153).

Внутри длинной серии обе цифры 4 и 3 приняты в записи 123454321 при score≈0,517.

## Как интерпретировать подборку

Тематические группы подобраны для изучения, не являются случайной выборкой и не оценивают частоту причин. Автоматические tags — пересекающиеся признаки текста. Разница в формулировке, пунктуации или LaTeX может сохранять смысл, но это не доказано сравнением одного токена. Для дообучения нужны отдельные train/validation/test по целым задачам; уже просмотренный heldout нельзя снова назвать слепым тестом.

Скрипт: `bash scripts/report/mtp_examples.sh`. JSON.gz и его SHA-256 записываются рядом с HTML. Параметры и локальный запуск описаны в [руководстве](../mtp-examples-guide.md).
