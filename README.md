# 计算机程序的构造和解释
在程序设计中，需要处理两类要素：***过程和数据***。

**数据** 是我们希望操作的东西, 而 **过程** 就是有关操作这些数据规则的描述。

Lisp 具有将 **过程** 表示为 **数据** 的能力。

## 构造过程抽象

### 程序设计的基本元素
每一种强有力的语言都提供三种机制:
1. **基本表达式**: 用于表示语言所关心的 **最简单的个体**。
2. **组合的方法**: 通过它们可以从较简单的东西出发构造出 **复合** 的元素。
3. **抽象的方法**: 通过它可以对 **复合对象命名**，并将它当作 **单元** 去操作。

#### 命名和环境
程序设计必不可少的一个方面，就是它需要提供一种通过命名去使用计算对象的方式。一般将名称标识符称为 ***变量*** ，它的值就是它所对应的 **对象**。

```lisp
;;该语句会使 lisp 解释器将数值 2 和 size 这个变量名称相关联，后续我们就可以通过 size 去引用数值 2 了。
(define size 2)
;; 等同于 5 * 2
(* 5 size)
;;一个 Lisp 程序总是由一大批相对简单的过程组成。
```
可以看到,Lisp解释器能将**值**与**符号**关联,而后又能提取出这些值,这意味着解释器必须维护某种存储能力,以便保持值和变量之间的映射关系。这种存储被称为 **环境**。

#### 组合式求值
**解释器** 本身按照下面 **过程** 工作:
1. 求值该组合式的各个子表达式。
2. 将作为最左子表达式(运算符)的值的那个 **过程**，应用于相应的 **实际参数(即: 子表达式运算对象的值)** 。

为了实现一个组合式的求值过程，必须对组合式里的每个元素执行同样的求值过程。在性质上，这一过程是 **递归** 的。

采用递归的思想可以很简洁的描述深度嵌套的情况，例如:
```lisp
(* (+ 2 (* 4 6))
   (+ 3 5 7) )
```
从上观察可知,反复的利用第一个步骤,总是可以把我们带到求值中的某一点,在这里遇到的不是 **组合式** 而是 **基本表达式** 。处理这些基础情况的规则如下:
1. 数的值就是它所表达的值
2. 内部运算符的值就是能完成相应操作的机器指令序列。
3. 其他名字的值就是在环境中关联于这一名字的那个对象。

我们可以将 **第二种规则** 看作是 **第三种规则** 的 **特殊情况**，为此只需要将 ***加 + 减 — 乘 * 除 /*** 这些运算符也存放在 **全局坏镜** 里，并将相应的指令序列作为与之关联的值。
在这里 **环境** 所扮演的角色就是 ***用于确定表达式各个符号的意义*** 。

对各个表达式的求值规则可以描述为一个简单的通用的规则和一组针对不多的形式的专门规则。

#### 复合规则
```lisp
;;过程定义
(define (square x) (* x x))
;; x^2 + y^2
(define (sum-of-square x y)
        (+ (square x) (square y)))
;; 3^2 + 4^2 => 9+16 = 25        
(sum-of-square 3 4)        
```

#### 过程应用的代换模型
求值一个 **组合式(运算符为复合过程的名称)** 解释器的计算过程可以采用两种模式:
1. **应用序求值**
应用序首先对 **运算符** 和 **各个运算对象** 求值，而后将得到的过程应用于得到的实际参数。
2. **正则序求值**
先不求出运算对象的值，直到实际需要他们的值时在做。采用这种求值方式，首先用 **运算表达式** 去 **代换形式参数** ,直至得到一个只包含 **基本运算符的表达式** 时，然后在去执行求值。

```lisp
;;复合过程定义
(define (f a)
        (sum-of-square (+ a 1) (* a 2)))
;;应用序求值过程
(f 5)
(sum-of-square (+ 5 1) (* 5 2))
(+ (square 6) (square 10))
(+ (* 6 6) (* 10 10))
(+ 36 100)
(136)
;;正则序求值过程
(f 5)
(sum-of-square (+ 5 1) (* 5 2))
(+ (square (+ 5 1)) (square (* 5 2)))
(+ (* (+ 5 1) (+ 5 1)) (* (* 5 2) (* 5 2)))
(+ (* 6 6) (* 10 10))
(+ 36 100)
(136)
;;判断程序是正则序求值还是应用序求值
(define (p) (p))
(define (test x y)
    (if (= 0 x) 
        0 
        y))
(test 0 (p))        
```
**Lisp 解释器** 在内部采用了 **应用序求值** 的方式,部分原因是为了避免对表达式的重复求值，以 **提升效率**。
更重要的是,在超出可采用替换方式模拟的过程范围之后,应用 **正则序求值方式会复杂很多** 。

**条件表达式和谓词**
```lisp
;;条件表达式
(define (abs x)
        (cond ((> x 0) x)
              ((= x 0) 0)
              ((< x 0) -x)))
;;谓词
(define (abs x)
        (if (< x 0)
        (- x)
        x))
```

#### 数学函数和程序过程之间的差异
**数学函数** 和 **程序过程** 之间的矛盾主要体现在描述一件事情的特征，与描述这件事如何去做之间普遍性差异的一个具体反应。

在数学中人们通常关心的是 **说明性描述** (是什么)，而在计算机程序中，人们则通常关心 **行动性描述** (怎么做)。

例如: 下面数学中 **平方根的函数定义**，我们虽然可以利用它去判断某个数是否为令一个数的平方根，或者根据它的定义，推倒出一些有关平方根的一般性事实。
但是这一数学函数定义 **并没有描述一个计算过程**，我们无法基于这个 **数学函数** 定义，在给定一个数之后，实际地找到这个数的 **平方根** 。

```math
\sqrt{x} = y ; y>=0 , {y^2} =x
```

一个程序过程就是一种 **模式**，它描述了一个 **计算过程的局部演化方式**，描述了这一计算过程中的每个步骤是怎样基于前面的步骤建立起来的。

程序过程的 **局部参数名** 必须局部于有关的过程体。过程的 **形式参数** 的具体名称是什么，完全不影响过程的运行结果。这样的名字被称为 **约束变量**。

```lisp
;;下面这两个过程定义是完全等价，无法区分的。
(define (square x) (* x x))
(define (square y) (* y y))
```
#### 线性递归和迭代
考虑由下面表达式定义的阶乘函数：

**n! = n * (n-1) * (n-2) ... * 3 * 2 * 1**

**n! = n * [(n-1) * (n-2) ... * 3 * 2 * 1] = n * (n-1)!**

程序过程的等价描述为:
```lisp
(define (factor n) 
    (if (= n 1) 
        1
        (* n (factor(- n 1)))))
 
;;实际运算过程如下：
;;(factor 5)
;;(* 5 (factor 4)
;;(* 5 (4 * (factor 3)))
;;(* 5 (x 4 (* 3 (* 2 (factor 1))))
;;(* 5 (* 4 (* 3 (* 2 1)))) 
```
另一种思路是将计算阶乘 n! 的规则描述为：先乘 1 和 2 ，而后将得到的结果乘以 3 这样一直下去直到达到 n。 
形式地说，维持一个变动中的乘积 product 以及一个从 1 到 n 的计数器 counter，计算过程可以描述为 counter  和 product ：

```lisp
;; product <- counter * product;
;; counter <- counter + 1;
(define (factor n) 
    (fact-iter 1 1 n)) 

(define (fact-iter product counter max-count) 
    (if (> counter max-count) 
        product 
        (fact-iter (* counter product) (+ counter 1) max-count)))
 
 ;; 实际运算过程如下：
;;(factor 6)
;;(iter 1 1 6)
;;(iter 1 2 6)
;;(iter 2 3 6)
;;(iter 6 4 6)
;;(iter 24 5 6)
;;(iter 120 6 6)
;;(iter 720 7 6)
;;=>720
```
#### 树形递归
斐波那契数列数由下面规则定义:
```math
Flib(n) = \begin{cases}
   0 &\text{if } n = 0 \\
   1 &\text{if } n = 1 \\
   Flib(n-1) + Flib(n-2) &\text{otherwise}
\end{cases}
```

可以将这个定义翻译为一个计算 **斐波那契数** 的递归过程：

```lisp
(define (fib n) 
    (cond ((= n 0) 0) 
            ((= n 1) 1) 
            (else (+ (fib (- n 1)) 
                       (fib (- n 2))))))
```

一般说，**树形递归** 计算过程里所需的 **步骤数正比于树中的节点数**，**空间需求正比于树的最大深度**。

与计算阶乘类似，可以规划出一种计算 **斐波那契数** 的 **迭代计算过程**，基本思路使用一对整数 ***a*** 和 ***b***，将它们分别初始化为 ***Fib(1) = 1*** 和 ***Fib(0) = 0*** , 而后反复地同时使用下面变换规则：

***a <- a + b
b <- a***

以迭代方式计算斐波那契数：

```lisp
(define (fib n)
    (fib-iter 1 0 n)) 

(define (fib-iter a b count) 
    (if (= count 0)
        b 
        (fib-iter (+ a b) a (- count 1))))
```
#### 实例: 换零钱方式的统计
给定任意数量的现金，写出一个程序过程，计算出所有换零钱方式的数量。

如果采用递归过程，这个问题有一种会很简单的解法，假设考虑的可用硬币类型种类排了某种顺序，于是就有以下关系：

将总数为 a 的现金换成 n 种硬币的不同方式的数目等于
* 将现金数 a 换成除第一种硬币之外的所有其他硬币的不同方式数目
* 将现金数 a - d 换成所有种类的硬币的不同方式数目，其中的 d 是第一种硬币的币值。

换成零钱的全部方式的数目，就等于完全不用第一种硬币的方式的数目，加上用了第一种硬币的换零钱方式的数目。

```lisp
(define (count-change amount)
    (cc amount 5))
    
(define (cc amount kinds-of-coins)
    (cond ((= amount 0) 1)
            ((or (< amount 0) (= kinds-of-coins 0)) 0)
            (else (+ (cc amount
                            (- kinds-of-coins 1))
                       (cc (- amount
                                (first-denomination kinds-of-coins))
                            kinds-of-coins)))))
                            
(define (first-denomination kinds-of-coins)
    (cond ((= kinds-of-coins 1) 1)
            ((= kinds-of-coins 2) 5)
            ((= kinds-of-coins 3) 10)
            ((= kinds-of-coins 4) 25)
            ((= kinds-of-coins 5) 50)))
            
 (count-change 100)
 ;; 292
```
不同的计算过程在消耗计算资源的速率上可能存在着巨大差异。描述这种差异的一种方便方式是用 **增长阶的记法**。

#### 高阶过程抽象

过程本身也是一类抽象，它描述了一些对于数的复合操作，但又不依赖于特定的数。

```lisp
;; 例如该过程并不是某个特定数值的立方，而是对任意的数得到其立方的方法
(define (cube x) (* x x x))
```
功能强大的程序设计语言有一个必然要求，就是能为公共的模式命名，建立抽象，而后在抽象层次上工作。过程提供了这种能力。因此除了简单的程序语言外，其他语言都包含定义过程机制的原因。

如果将过程限制为只能以数作为参数，那也会严重限制建立抽象的能力。以过程作为参数，或者以过程作为返回值。这类操作过程的过程称为高阶过程。

#### 过程作为参数
```lisp
;; 示例一: 计算从 a 到 b 的各整数之和
(define (sum-integers a b) 
    (if (> a b) 
        0 
        (+ a (sum-integers (+ a 1) b))))
;; 示例二: 计算给定范围内的整数的立方之和
(define (sum-cubes a b) 
    (if (> a b) 
        0 
        (+ (cube a) (sum-cubes (+ a 1) b))))
;; 示例三：计算下面的序列之和
(define (pi-sum a b) 
    (if (> a b) 
        0 
        (+ (/ 1.0 (* a (+ a 2))) (pi-sum (+ a 4) b))))
```
可以明显看出，以上三个过程共享着一个 **公共的基础模式**。

```lisp
(define (<name> a b) 
    (if (> a b) 
        0 
        (+ (<term> a) 
            (<name> (<next> a) b))))
```
该定义类比与数学中的 **求和记法** :

```math
\sum_{n=a}^b f(a) + ... + f(b)
```

作为程序模式，我们也希望所用的语言足够强大，能用于写出一个过程，去表述求和的概念，而不是只能写计算特定求和的过程。

```lisp
;; 只需要按照上面给出的模式，将空位翻译为形式参数
(define (sum term a next b)
    (if (> a b) 
        0 
        (+ (term a) 
            (sum term (next a) next b))))
 ;; 计算给定序列的立方和
(define (inc n) (+ n 1)) 

(define (sum-cubes a b) 
    (sum cube a inc b))
;; 计算数字 1 到 10 的立方和    
(sum-cubes 1 10)    
;;=> 3025
;; 计算数字 1 到 10 整数之和
(define (identity x) x) 
(define (sum-integers a b) 
    (sum identity a inc b))
;;=> 55
```
#### 用 lambda 构造过程
在上面 ***sum*** 时，必须显示定义一些如 ***inc*** 一类的简单函数，以便用它们作为 **高阶函数的参数**，这些有些冗余的做法，可以通过引入一种 ***lambda*** 特殊形式完成描述。

```lisp
(lambda (x) (+ x 4))
(lambda (x) (/ 1.0 (* x (+ x 2))))
;; 直接描述 pi-sum ，无须定义任何辅助过程。
(define (pi-sum a b)
    (sum (lambda (x) (/ 1.0 (* x (+ x 2)))) 
        a 
        (lambda (x) (+ x 4)) b))
```

一般而言，除了不为有关过程提供名字外，***lambda*** 用和 ***define*** 同样的方式创建过程。

```lisp
(lambda (<formal-parameters>) <body>)
;; 以下两种过程定义等价
(define (plus4 x) (+ x 4))
(define plus4 (lambda (x) (+ x 4)))
```
#### 复合过程求解函数零点和不动点
**区间折半法** 是寻找方程 ***f(x) = 0*** 一种简单而又强有力的方法，这里的 ***f*** 是一个 **连续函数**。
这种方法的基本思路是： 如果给定点 ***a*** 和 ***b***  有 ***f(a) < 0 < f(b)*** 那么 ***f*** 在 ***a*** 和 ***b*** 之间 **必然有一个零点**。
为了确定这个 **零点**， 令 ***x*** 是 ***a*** 和 ***b*** 的平均值并计算出 ***f(x)*** 。
如果 ***f(x) > 0*** 那么在 ***a*** 和 ***x*** 之间必然有一个 ***f*** 的 **零点**；
如果 ***f(x) <0*** , 那么在 ***x*** 和 ***b*** 之间必然有一个 ***f*** 的 **零点**;
持续下去，就能 **确定越来越小的区间**，且保证在其中必然有 ***f*** 的一个 **零点**。
当区间足够小时就结束这一计算过程。

```lisp
(define (search f neg-point pos-point)
    (let ((midpoint (average neg-point pos-point))) 
    (if (close-enough? neg-point pos-point) 
        midpoint 
        (let ((test-value (f midpoint))) 
        (cond ((positive? test-value) 
        (search f neg-point midpoint)) 
        ((negative? test-value) 
        (search f midpoint pos-point)) 
        (else midpoint))))))

(define (half-interval-method f a b) 
    (let ((a-value (f a)) 
          (b-value (f b))) 
      (cond ((and (negative? a-value) (positive? b-value)) 
                (search f a b)) 
               ((and (negative? b-value) (positive? a-value)) 
                (search f b a)) 
                (else (error "Values are not of opposite sign" a b)))))
```
#### 函数的不动点
数 ***x*** 称为函数 ***f*** 的不动点，如果 ***x*** 满足方程 ***f(x) = x*** 。对于某些函数，通过从某个初始猜测出发，反复地应用 ***f*** ；

```lisp
(define tolerance 0.00001) 

(define (fixed-point f first-guess) 
    (define (close-enough? v1 v2) 
        (< (abs (- v1 v2)) tolerance)) 
    (define (try guess) 
        (let ((next (f guess))) 
            (if (close-enough? guess next) 
                next 
                (try next)))) 
        (try first-guess))
```


#### 过程作为返回值
将 **过程作为参数传递** 能够显著 **增强程序设计语言的表达能力**。通过创建另一种其返回值也是过程的过程，我们还能得到进一步的表达能力。

```lisp
(define (average-damp f) 
    (lambda (x) (average x (f x))))

((average-damp square) 10)
;; => 55

(define (fixed-point-of-transform g transform guess) 
    (fixed-point (transform g) guess))
 
(define (sqrt x) 
    (fixed-point-of-transform (lambda (y) (/ x y)) 
                                      average-damp 1.0))

(define (sqrt x) 
    (fixed-point-of-transform (lambda (y) (- (square y) x)) 
                                      newton-transform 1.0))
```
## 构造数据抽象
许多程序在设计时就是为了模拟复杂的现象，因此常常需要构造一些计算对象，这些对象都是由一些部分组成的，以便去模拟真实世界里那些具有若干侧面的现象。

将数据对象组合起来，形成复合数据的方式。可以提升设计程序时所位于的概念层次，提高设计的模块性，增强语言的表达能力。使得程序设计可以在比语言提供的基本数据对象更高的概念层次上，处理与数据有关的各种问题。

**数据抽象** 是一种方法学, 它能将一个复合数据对象的使用，与该数据对象怎样由更基本的数据对象构造起来的细节隔离开；

**数据抽象** 的基本思想，就是设法构造出一些使用 **复合数据对象** 的程序，使程序就像是在抽象数据上操作一样。

程序中 **使用数据的方式** 应该是这样的，**除了完成当前工作所必要的东西之外，它们不对所用数据做任何假设** ；
一种 "具体" 数据表示的定义，应该与程序中使用数据的方式无关;

这两个部分之间的界面将是一组过程，称为 **选择函数 和 构造函数** ；

#### 实例 ：有理数的算数运算
有理数定义：

可以将这些规则表述为如下几个过程:
```lisp
;; 有理数加法
(define (add-rat x y) 
    (make-rat (+ (* (numer x) (denom y))
                      (* (numer y) (denom x))) 
                      (* (denom x) (denom y)))) 
;; 有理数减法
(define (sub-rat x y)
    (make-rat (- (* (numer x) (denom y)) 
                     (* (numer y) (denom x))) 
                     (* (denom x) (denom y)))) 
;; 有理数乘法
(define (mul-rat x y) 
    (make-rat (* (numer x) (numer y)) 
                  (* (denom x) (denom y))))
;; 有理数除法
(define (div-rat x y) 
    (make-rat (* (numer x) (denom y)) 
                  (* (denom x) (numer y))))

(define (equal-rat? x y) 
    (= (* (numer x) (denom y)) 
        (* (numer y) (denom x))))
```
#### Pairs 序对
Scheme Lisp 提供了一种称为 **Pairs** 的复合结构，这种结构可以通过基本过程 ***cons*** 构造出来。过程 ***cons*** 取两个参数，返回一个包含这两个参数作为其成分的 **复合数据对象** 。

```lisp
(define x (cons 1 2)) 
(car x) 
;;=> 1 
(cdr x) 
;;=> 2

(define x (cons 1 2)) 
(define y (cons 3 4)) 
(define z (cons x y)) 

(car (car z)) 
;; =>1 
(car (cdr z)) 
;;=>3
```
#### 有理数的表示
```lisp
;; 有理数的定义
(define (make-rat n d) (cons n d)) 
(define (numer x) (car x)) 
(define (denom x) (cdr x))
;; 打印有理数
(define (print-rat x) 
    (newline) 
    (display (numer x)) 
    (display "/") 
    (display (denom x)))
 
(define one-half (make-rat 1 2)) 

(print-rat one-half) 
;; =>1/2 
(define one-third (make-rat 1 3)) 
(print-rat (add-rat one-half one-third)) 
;;=>5/6 
(print-rat (mul-rat one-half one-third)) 
;;=>1/6 
(print-rat (add-rat one-third one-third)) 
;;=>6/9
```
一般而言，**数据抽象的基本思想** 就是为每一类数据对象标识出一组操作，使得对这类数据对象得所有操作都可以基于它们表述，而且在操作这些数据对象时也只使用它们。

**抽象屏障** 隔离了系统中不同的层次。在每一层上，这种屏障都把使用 **数据抽象** 的程序与 **实现数据抽象的程序** 分开 。

每一层次中的过程构成了所定义的抽象屏障的界面，联系起系统中的不同层次。

**抽象屏障** 这以简单的思想有许多优点。第一个优点是这种方法使程序很容易维护和修改。任意一种 **比较复杂的数据结构**，都可以 **以多种不同方式用程序设计语言所提供的基本数据结构表示**。

**缺点** 是 **表示方式改变** 了，所有 **受影响的程序** 也都需要随之改变。


一般而言，总是可以将 **数据定义为一组适当的 选择函数和构造函数**，以及为使这些 **过程成为一套合法表示**，它们就 **必须满足的一组特定条件** 。

只使用过程就可以实现 **Pairs** ，下面使有关的定义：

```lisp
(define (cons x y) 
    (define (dispatch m) 
        (cond ((= m 0) x) 
                 ((= m 1) y)
                 (else (error "Argument not 0 or 1 -- CONS" m)))) dispatch) 
                 
(define (car z) (z 0)) 
(define (cdr z) (z 1))

;; lambda 定义 cons
(define (lambda-cons x y)
  (lambda (m) (m x y)))
  
(define (lambda-car z)
  (z (lambda (p q) p)))
  
(define (lambda-cdr z)
  (z (lambda (p q) q)))
;; 测试
(define z (lambda-cons 10 2))
(lambda-car z)
;;=> 10
(lambda-cdr z)
;;=> 2
```
**数据的过程性表示** 将在程序设计里扮演一种 **核心角色**。

```lisp
;; church 计数
;;Lambda 演算是一个完全用 Lambda 进行编程的正式系统。 事实证明它非常强大，但更像是一种学术/启发性的练习。
(define zero
  (lambda (f)
    (lambda (x) x)))

(define one 
    (lambda (f) 
        (lambda (x) (f x))))

(define two
    (lambda (f)
        (lambda (x)
            (f (f x)))))

(define (add-one n)
  (lambda (f)
    (lambda (x)
      (f ((n f) x)))))
      
;; f 是要应用 n 次的函数, x 是要操作的项。 这两个 lambda 可以组合,但正式的 lambda 演算规定只能使用单个参数(当然,多个参数可以用 lambda 表示)
;;=> 4
(lambda (f)
  (lambda (x)
    (f (f (f (f x))))))

;;=> 0
(lambda (f)
  (lambda (x) x))  
  
;; 测试
(define (add-number x) (+ 1 x))
((one add-number) 5)
;; => 6
```
#### 层次性数据和闭包性质
某种 **组合数据对象** 的 **操作** 满足 **闭包性质**，那就是说，通过它组合起数据对象得到的结果本身还可以通过同样的操作再进行组合。

```lisp
;; 序列闭包特性
(cons (cons 1 2)
        (cons 4 5))
        
(cons (cons 1 (cons 2 3))
        4)
```
#### list 序列的表示
利用 **序对** 可以构造出的一类有用结构是 **list 序列** 一批数据对象的一种有序汇集。

```lisp
(cons 1 
    (cons 2 
        (cons 3 
            (cons 4 nil))))
;; 等价于
(list 1 2 3 4)

(define one-through-four (list 1 2 3 4))
;; lisp 的操作
(car one-through-four)
;;=>1
(cdr one-through-four)
;;=>(2 3 4)
(car (cdr one-through-four))
;;=>2
(cons 10 one-through-four)
;;=>(10 1 2 3 4)
(cons 5 one-through-four)
;;=>(5 1 2 3 4)
```
#### list 序列的映射
一个特别有用的操作是将某种变换应用于一个表的所有元素，得到所有结果构成的表。
```lisp
(define (scale-list items factor) 
    (if (null? items) 
        nil 
        (cons (* (car items) factor) 
                (scale-list (cdr items) factor)))) 

(scale-list (list 1 2 3 4 5) 10) 
;;=> (10 20 30 40 50)
;; map 等价实现
(define (scale-list items factor) 
    (map (lambda (x) (* x factor)) 
            items))
;; 将公共模式表示为一个高阶过程，它有一个过程参数和序列参数，返回将这一过程应用于表中各个元素得到的结果形成的表
(define (map proc items) 
    (if (null? items) 
        nil 
        (cons (proc (car items)) 
                (map proc (cdr items))))) 
;; map 高阶过程抽象        
(map abs (list -10 2.5 -11.6 17)) 
;;=> (10 2.5 11.6 17) 
(map (lambda (x) (* x x)) (list 1 2 3 4)) 
;;=> (1 4 9 16)

 ;; foreach 高阶抽象
 (define (for-each proc items)
  (define (iter lst result)
    (if (null? lst)
        #t
        (iter (cdr lst)
              (proc (car lst)))))
  (iter items (car items)))

(for-each (lambda (x) (newline) (display x))
    (list 57 321 88))
 ;;=> 57
 ;;=> 321
 ;;=> 88#t
```
***map*** 建立起了一层抽象屏障，将实现 **表变换的过程的实现**，与如何 **提取表中元素** 以及 **组合结果** 的细节隔离开；

```lisp
(define (square x) (* x x))
;;将 map 高阶函数 的线性递归实现过程改为迭代过程
(define (square-list items)
 (define (iter things answer)
   (display "iter")
   (display things)
   (display answer)
   (newline)
   (if (null? things)
       answer
       (iter (cdr things) 
             (cons answer
                   (square (car things)))
             )))
  (iter items null))
  
(square-list (list 1 2 3 4))
;;=> iter(1 2 3 4)()
;;=> iter(2 3 4)(() . 1)
;;=> iter(3 4)((() . 1) . 4)
;;=> iter(4)(((() . 1) . 4) . 9)
;;=> iter()((((() . 1) . 4) . 9) . 16)
(define (square-list-two items)
 (define (iter things answer)
   (display "iter")
   (display things)
   (display answer)
   (newline)
   (if (null? things)
       answer
       (iter (cdr things) 
             (cons (square (car things))
                   answer )
             )))
  (iter items null))

(square-list-two (list 1 2 3 4))
;;=> iter(1 2 3 4)()
;;=> iter(2 3 4)(1)
;;=> iter(3 4)(4 1)
;;=> iter(4)(9 4 1)
;;=> iter()(16 9 4 1)
```
#### 层次性结构(Tree)
```lisp
;; 用 lisp 列表构建出树
(define x (cons (list 1 2) (list 3 4)))
(length x)
;;=> 3
;; 计算树形结构中的叶子节点
(define (count-leaves x)
     (cond ((null? x) 0) 
           ((not (pair? x)) 1)
           (else (+ (count-leaves (car x))
                      (count-leaves (cdr x))))))
;; 叶子节点总数         
(count-leaves x)
;;=> 4
(list x x)
;;=> (((1 2) 3 4) ((1 2) 3 4))
(length (list x x))
;;=> 2
(count-leaves (list x x))
;;=> 8
;;层次结构的 map 映射操作
(define (scale-tree tree factor)
    (cond ((null? tree) nil) 
            ((not (pair? tree)) (* tree factor)) 
            (else (cons (scale-tree (car tree) factor) 
                           (scale-tree (cdr tree) factor))))) 
;; 测试
(scale-tree (list 1 (list 2 (list 3 4) 5) (list 6 7)) 10)
;;=> (10 (20 (30 40) 50) (60 70))
```
#### 序列作为一种通用约定[接口]
 对 **复合数据** 的 **数据抽象** 能在工作中帮助我们设计出 **不被数据表示的细节纠缠的程序**，使程序能够保持很好的 **弹性**。

通过实现为 **高阶过程的程序抽象**，抓住 **处理数值数据** 的一些 **程序模式**。
要在 **复合数据** 上工作做出类似的操作，则对 **操控数据结构的方式** 有着 **深刻的依赖性** 。

```lisp
;;迭代出一棵树的叶子节点
;;过滤叶子节点，选出其中的奇数
;;对选出的每一个数求平方
;;用 + 累积起得到的结果从 0 开始
(define (sum-odd-squares tree) 
    (cond ((null? tree) 0) 
    ((not (pair? tree)) 
     (if (odd? tree) (square tree) 0))
     (else (+ (sum-odd-squares (car tree)) 
                (sum-odd-squares (cdr tree))))))

;;迭代从 0 到 n 的整数
;;对每个整数计算相应的斐波那契数
;;过滤整数，选出其中的偶数
;;用cons积累得到的结果，从空表开始
(define (even-fibs n) 
    (define (next k) 
        (if (> k n) 
            nil 
            (let ((f (fib k))) 
                (if (even? f) 
                    (cons f (next (+ k 1))) 
                    (next (+ k 1)))))) 
    (next 0))
```
在这两个过程里，采用不同的方式分解了这个计算，将迭代工作散布在程序中各处，并将它与 **映射、过滤器和累计器混** 在一起。

如果能够重新组织这个程序，用类似于信号流结构明显表现在写出的过程中，将会大大 **提高结果代码的清晰性**。

要组织好这些程序，使它们能够更清晰地反应上面信号流的结构，最关键的一点就是将 **注意力集中在** 处理过程中从一个步骤流向下一个步骤的 **信号**。

```lisp
(map square (list 1 2 3 4 5))
;; => (1 4 9 16 25)

;;过滤一个序列
(define (filter predicate sequence) 
    (cond ((null? sequence) nil) 
            ((predicate (car sequence)) 
             (cons (car sequence) 
             (filter predicate (cdr sequence)))) 
             (else (filter predicate (cdr sequence)))))

(filter odd? (list 1 2 3 4 5))
;; => (1 3 5)

;;累加器
(define (accumulate op initial sequence) 
    (if (null? sequence)
        initial 
        (op (car sequence) 
        (accumulate op initial (cdr sequence)))))

(accumulate + 0 (list 1 2 3 4 5))
;; => 15 
 (accumulate cons nil (list 1 2 3 4 5))
;;=> (1 2 3 4 5)

;;实现信号流图
(define (enumerate-interval low high)
    (if (> low high) 
        nil 
        (cons low (enumerate-interval (+ low 1) high)))) 
        
(enumerate-interval 2 7)
;; => (2 3 4 5 6 7)

;;树形结构叶子节点遍历
(define (enumerate-tree tree) 
    (cond ((null? tree) nil) 
    ((not (pair? tree)) (list tree)) 
    (else (append (enumerate-tree (car tree)) 
                            (enumerate-tree (cdr tree))))))

(enumerate-tree (list 1 (list 2 (list 3 4)) 5))
;; => (1 2 3 4 5)
;;将上面的过程组合成复合数据抽象
(define (sum-odd-squares tree) 
    (accumulate + 
                     0 
                     (map square 
                            (filter odd? 
                                    (enumerate-tree tree)))))

(define (even-fibs n) 
  (accumulate cons 
                    nil 
                    (filter even? 
                            (map fib 
                                    (enumerate-interval 0 n)))))
```
将程序表示为一些针对 **序列的操作**，这样做的价值就在于能帮助我们得到 **模块化的程序设计**，得到由一些比较独立的片段的组合构成的设计。通过提供一个标准部件的库。
并使这些部件都有着一些能以各种灵活方式相互连接的 **通用约定**，将能进一步 **推进** 做 **模块化程序设计**。

在工程设计中，**模块化结构** 是 **控制复杂性** 的一种威力 **强大的策略**。

程序设计的一个关键概念是 **分层设计** 问题。
这一概念说的是一个 **复杂系统应该通过一系列的层次构造出来**，为了描述这些层次，需要使用一系列的语言。
构造各个 **层次的方式**，就是 **设法组合** 起作为这一层次中部件的 **各种基本元素**，而这样构造出的部件又可以作为另一个层次里的基本元素。
在 **分层设计** 中，每个层次上所用的语言都提供了一些 ***基本元素、组合手段***，还有对该层次中的 ***适当细节做抽象*** 的手段。

#### 嵌套映射
可以扩充 list 泛型，将许多通常用 **嵌套循环** 表述得计算包含进来。比如以下问题：
***给定自然数 n 找出所有不同得有序对 i 和 j，其中 1 <= j < i <= n, 使得 i + j 是素数*** ; 假定 n 是 6，满足条件得序对就是：

```
\def\arraystretch{1.5}
   \begin{array}{c:c}
   i & 2 & 3 & 4 & 4 & 5 & 6 & 6 \\ \hline
   j & 1 & 2 & 1 & 3 & 2 & 1 & 5\\ 
   \hdashline
   i+j & 3 & 5 & 5 & 7 & 7 & 7 & 11
\end{array}
```

这个计算一种很自然得解决方案是: 首先生成出所有小于等于 n 的正整数有序对，而后通过过滤，得到那些和为素数的有序对；
最后对每个通过了过滤的 **序对 (i,j)** , 产生出一个 **三元组 (i,j,i+j)**


```lisp
(accumulate append 
                nil 
                (map (lambda (i) 
                        (map (lambda (j) (list i j)) 
                                (enumerate-interval 1 (- i 1)))) 
                        (enumerate-interval 1 n)))
 
(define (flatmap proc seq) 
    (accumulate append nil (map proc seq)))

(define (prime-sum? pair) 
    (prime? (+ (car pair) (cadr pair))))

(define (make-pair-sum pair)
    (list (car pair) (cadr pair) (+ (car pair) (cadr pair))))

(define (prime-sum-pairs n) 
    (map make-pair-sum 
        (filter prime-sum? 
            (flatmap 
            (lambda (i) 
                (map (lambda (j) (list i j)) 
                    (enumerate-interval 1 (- i 1)))) 
                (enumerate-interval 1 n)))))

(define (permutations s) 
    (if (null? s) ; empty set? 
        (list nil) ; sequence containing empty set 
        (flatmap (lambda (x) 
                        (map (lambda (p) (cons x p)) 
                                (permutations (remove x s)))) 
                     s)))

(define (remove item sequence) 
    (filter (lambda (x) (not (= x item))) 
            sequence)) 
```
#### 八皇后谜题
**八皇后谜题** 是指怎样将八个皇后摆在国际象棋盘上，使得任意一个皇后都不能攻击另一个皇后 (任意两个皇后都不能在同一行，同一列或者同一对角线上)。

一种解决方法按一个方向处理棋盘，每次在每一列里放一个皇后。
如果现在已经放好了 ***k - 1*** 个皇后，**第 k 个皇后** 就必须放在不会被已在棋盘上的任何皇后攻击的位置上。

用递归描述这一过程 : 假定已经生成了在棋盘的前 ***k - 1 列*** 中放置 ***k - 1 个皇后*** 的所有可能方式，现在需要的就是对于其中的每种方式，生成出将下一个皇后放在 ***第 k 列*** 中每一行的 **扩充集合** 。
而后过滤它们，只留下能使位于 ***第 k 列*** 的皇后与其他皇后相安无事的那些扩充。
这样就能产生出将 k 个皇后放置在 **前 k 列** 的所有格局的序列。
继续这一过程，将能产生出这一个 **谜题的所有解**，而不是一个解。

```lisp
(define (queens board-size)
    (define (queen-cols k)
        (if (= k 0)
            (list empty-board)
            (filter
            (lambda (positions) (safe? k positions))
            (flatmap
            (lambda (rest-of-queens)
            (map (lambda (new-row)
                    (adjoin-position
                     new-row k rest-of-queens))
                     (enumerate-interval 1 board-size)))
             (queen-cols (- k 1))))))
         (queen-cols board-size))
                         
(define (adjoin-position row col board)
    (if (null? board)
        (list (list col row))
        (append (list (list col row)) board)))
        
(define empty-board '())

(define (safe? k sets)
    (define (bad-pairs? pair1 pair2)
        (let ((x (abs (- (car pair1) (car pair2))))
              (y (abs (- (cadr pair1) (cadr pair2)))))
        (or (= (cadr pair1) (cadr pair2))
             (= x y))))
             
    (define (safe-iter pair seq)
        (cond ((null? seq) #t)
                ((bad-pairs? pair (car seq)) #f)
                (else (safe-iter pair (cdr seq)))))
    (if (null? sets)
        #t
        (safe-iter (car sets) (cdr sets))))
```

#### 霍夫曼编码树

```lisp
; leaf
(define (make-leaf symbol weight)
  (list 'leaf symbol weight))

(define (leaf? object)
  (eq? (car object) 'leaf))

(define (symbol-leaf object)
  (cadr object))

(define (weight-leaf object)
  (caddr object))
; test leaf
(define leaf (make-leaf 'A 1))
(eq? (leaf? leaf) true)
(eq? (symbol-leaf leaf) 'A)
(eq? (weight-leaf leaf) 1)

; code-tree
(define (make-code-tree left right)
  (list 
    left
    right
    (append (symbols left) (symbols right))
    (+ (weight left) (weight right))))

(define (left-branch tree) (car tree))

(define (right-branch tree) (cadr tree))

(define (symbols tree)
  (if (leaf? tree)
    (list (symbol-leaf tree))
    (caddr tree)))

(define (weight tree)
  (if (leaf? tree)
    (weight-leaf tree)
    (cadddr tree)))
; test tree
(define tree (make-code-tree (make-leaf 'A 1) (make-leaf 'B 2)))
(eq? (leaf? (left-branch tree)) true)
(eq? (symbol-leaf (right-branch tree)) 'B)
(and (eq? (car (symbols tree)) 'a) 
  (eq? (cadr (symbols tree)) 'b))

; adjoin
(define (adjoin-set x set)
  (cond ((null? set)
      (list x))
    ((> (weight (car set)) (weight x))
      (cons x set))
    (else (cons (car set) (adjoin-set x (cdr set))))))


(define (make-leaf-set pairs)
  (if (null? pairs)
    '()
    (let ((pair (car pairs)))
      (adjoin-set (make-leaf (car pair) (cadr pair))
        (make-leaf-set (cdr pairs))))))

(define (make-tree leaves)
  (cond ((or (null? (car leaves)) (null? (cadr leaves)))
      (error "leaves is not enough"))
    ((null? (cddr leaves))
      (make-code-tree (car leaves) (cadr leaves)))
    (else (make-tree (adjoin-set (make-code-tree (car leaves) (cadr leaves)) (cddr leaves))))))

; test make leaf set
(define leaf-set (make-leaf-set '((A 1) (B 3) (C 2))))
(and (eq? (caddar leaf-set) 1)
  (eq? (caddr (cadr leaf-set)) 2)
  (eq? (caddr (caddr leaf-set)) 3)
  (null? (cdddr leaf-set)))

; test make tree
(define tree (make-tree leaf-set))
(weight tree)
(eq? (weight tree) 6)

; encode
(define (encode tree)
  (define (visit n bits)
    (if (leaf? n)
      (cons (symbol-leaf n) bits)
      (cons (visit (left-branch n) (cons 0 bits))
        (visit (right-branch n) (cons 1 bits)))))
  (visit tree '()))

; test encode
(encode tree) ; outputs: ((b 0) (a 0 1) c 1 1)
```

#### 抽象数据的多重表示
**数据抽象** 屏障是控制复杂性的强有力工具。通过对数据对象基础表示的屏蔽，我们就可以将设计一个大程序的任务，分割为一组可以分别处理的较小任务。

然而，对于一个数据对象也可能存在多种有用的表示方式，而且我们也可能希望所设计的系统能处理多种表示形式。[一个现实的例子是: **复数** 就可以表示为两种几乎等价的模式 **直角坐标形式** 和 **极坐标形式**]

更现实的一种原因是: 一个程序系统通常是由许多人通过一个相当长时期的工作完成的，系统的需求也在随着时间而不断变化。在这样的环境中，要求每个人都在 **数据表示的选择上达成一致** 是根本不可能的事情。

因此，除了需要将表示与使用相隔离的 **数据抽象** 屏障之外，还需要有抽象屏蔽去隔离互不相同的设计选择，以便允许不同的设计选择在同一个程序里共存。

由于大型程序常常是通过组合起一些现存模块构造起来的，而这些模板又是独立设计的，需要一些方法 **[通用型过程]——也就是那种可以在不止一种数据表示上操作的过程** ，使程序可能逐步地将许多模块结合成一个大型系统，而不必去重新设计或者重新实现这些模块。

这里构造 **通用性过程** 所采用的主要技术，是让它们在带有 **类型标志** 的数据对象上工作。

##### **带类型标志的数据**

认识 **数据抽象** 的一种方式使将其看作 "**最小允诺原则**" 的一个应用。
由 **选择函数** 和 **构造函数** 形成的抽象屏障，可以把所用数据对象选择具体表示形式的事情尽量往后推, 而且还能保持系统设计的最大灵活性。
**最小允诺原则** 还能推进到更极端的情况。可以在设计完成 **选择函数** 和 **构造函数** 仍然维持所用表示方式的 **不确定性**。
完成这种区分的一种方式，就是在每个数据里包含一个 **类型标志** 部分, 借助于这个标志就可以确定应该使用的 **选择函数** 了。

##### **数据导向的程序设计和可加性**
检查一个 **数据项类型** ,并据此去调用某个适当过程称为 **基于类型的分派** 。
在系统设计中，这是一种获得 **模块性** 的强有力策略。

这样实现的 **分派** 由两个显著的 **弱点** :
1. 这些 **通用性界面** 过程必须知道所有的不同表示。
2. 即使这些独立的表示形式可以分别设计，也必须保证在整个系统里不存在两个名字相同的过程。

位于以上两个弱点之下的基础问题是，上面这种实现通用性界面的技术 **不具有可加性**。
每次新增一种 **新的表示形式** 时, 实现 **通用选择函数** 的人都必须修改他们的过程，而那些做独立的表示的界面人也必须修改其代码,以避免 **名称冲突** 问题。

一种称为 **数据导向的程序设计** 的编程技术提供了 **将系统设计进一步模块化** 的方法。
在需要处理的是针对不同类型的一集公共通用型操作时,我们正在处理一个 **二维表格**，其中一维包含所有可能的操作，另一个维度就是所有可能的类型。

**数据导向** 的程序设计就是一种使程序能直接利用这种表格工作的程序设计技术。

在设计大型系统时，处理好一大批 **相互有关的类型** 而同时又能 **保持模块性** ,这是一个非常困难的问题，也是当前正在继续研究的一个领域。

## 模块化、对象和状态