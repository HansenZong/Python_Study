
https://www.jianshu.com/p/a260a8c43e44

https://www.jianshu.com/p/83c8ef18a1e8


NumPy（Numerical Python）是Python语言的一个扩充程序库。支持高级大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。
ndarray类

  NumPy中的数组类被称为ndarray，要注意的是numpy.array与Python标准库中的array.array是不同的。ndarray具有如下比较重要的属性：
ndarray.ndim

  ndarray.ndim表示数组的维度。
ndarray.shape

  ndarray.shape是一个整型tuple，用来表示数组中的每个维度的大小。例如，对于一个n行和m列的矩阵，其shape为(n,m)。
ndarray.size

  ndarray.size表示数组中元素的个数，其值等于shape中所有整数的乘积。
ndarray.dtype

  ndarray.dtype用来描述数组中元素的类型，ndarray中的所有元素都必须是同一种类型，如果在构造数组时，传入的参数不是同一类型的，不同的类型将进行统一转化。除了标准的Python类型外，NumPy额外提供了一些自有的类型，如numpy.int32、numpy.int16以及numpy.float64等。
ndarray.itemsize

  ndarray.itemsize用于表示数组中每个元素的字节大小。

代码示例：
>>> import numpy as np
>>> a = np.arange(15).reshape(3,5)
>>> a
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
>>> a.shape
(3, 5)
>>> a.ndim
2
>>> a.dtype.name
'int64'
>>> a.dtype
dtype('int64')
>>> a.size
15
>>> a.itemsize
8
>>> type(a)
<class 'numpy.ndarray'>
>>> 
>>> b = np.array([1,2,3,4,5,6,7,8,9])
>>> b
array([1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> c = np.array([1,2,3,4,5,6,'7','a','b'])
>>> c
array(['1', '2', '3', '4', '5', '6', '7', 'a', 'b'], dtype='<U21')
>>> type(b)
<class 'numpy.ndarray'>
>>> type(c)
<class 'numpy.ndarray'>
>>> c.dtype
dtype('<U21')
>>> b.dtype
dtype('int64')
>>> c.itemsize
84
>>> b.itemsize
8

数组创建

  NumPy中创建数组的方式有若干种。最简单的，可以直接利用Python中常规的list和tuple进行创建。

>>> import numpy as np
>>> a = np.array([1,2,3,4,5,6])
>>> b = np.array((1,2,3,4,5,6))
>>> a
array([1, 2, 3, 4, 5, 6])
>>> b
array([1, 2, 3, 4, 5, 6])

  这里需要注意传入的参数，下面的第一种方式是错误的：

>>> a = np.array(1,2,3,4)    # WRONG
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: only 2 non-keyword arguments accepted
>>> a = np.array([1,2,3,4])  # RIGHT

  另外，传入的参数必须是同一结构,不是同一结构将发生转换。

>>> import numpy as np
>>> a = np.array([1,2,3.5])
>>> a
array([1. , 2. , 3.5])
>>> b = np.array([1,2,3])
>>> b
array([1, 2, 3])
>>> c = np.array(['1',2,3])
>>> c
array(['1', '2', '3'], dtype='<U1')
>>> 

  另外，array还可以将序列的序列转换成二位数组，可以将序列的序列的序列转换成三维数组，以此类推。

>>> import numpy as np
>>> a = np.array([[1,2,3],[2,3,4]])
>>> a
array([[1, 2, 3],
       [2, 3, 4]])
>>> b = np.array([[1,2,3],[2,3,4],[3,4,5]])
>>> b
array([[1, 2, 3],
       [2, 3, 4],
       [3, 4, 5]])
>>> 

  另外，创建数组的时候，可以明确的规定数组的类型。

>>> c = np.array([1,2,3], dtype = complex)
>>> c
array([1.+0.j, 2.+0.j, 3.+0.j])
>>> d = np.array([[1,2,3],[4,5,6]], dtype = '<U1')
>>> d
array([['1', '2', '3'],
       ['4', '5', '6']], dtype='<U1')
>>> 

  另外，NumPy还提供了便捷地创建特定数组的方式。

>>> import numpy as np
>>> a = np.zeros((3,4))
>>> a
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
>>> b = np.zeros((2,2,2))
>>> b
array([[[0., 0.],
        [0., 0.]],

       [[0., 0.],
        [0., 0.]]])
>>> c = np.ones((3,3))
>>> c
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])
>>> d = np.ones((3,3), dtype = np.int16)
>>> d
array([[1, 1, 1],
       [1, 1, 1],
       [1, 1, 1]], dtype=int16)
>>> e = np.arange(15)
>>> e
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])
>>> f = np.arange(15).reshape(3,5)
>>> f
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
>>> g = np.arange(0,15,3)
>>> g
array([ 0,  3,  6,  9, 12])
>>> h = np.arange(0,3,0.3)
>>> h
array([0. , 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7])

>>> from numpy import pi
>>> np.linspace( 0, 2, 9 )                 # 9 numbers from 0 to 2
array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
>>> x = np.linspace( 0, 2*pi, 100 )        # useful to evaluate function at lots of points
>>> f = np.sin(x)

基本操作

  对数组中的算术操作是元素对应（elementwise）的，例如，对两个数组进行加减乘除，其结果是对两个数组对一个位置上的数进行加减乘除，数组算术操作的结果会存放在一个新建的数组中。

>>> import numpy as np
>>> a = np.array([10,20,30,40])
>>> b = np.arange(4)
>>> a
array([10, 20, 30, 40])
>>> b
array([0, 1, 2, 3])
>>> c = a - b
>>> c
array([10, 19, 28, 37])
>>> a
array([10, 20, 30, 40])
>>> b
array([0, 1, 2, 3])
>>> b**2
array([0, 1, 4, 9])
>>> b
array([0, 1, 2, 3])
>>> a<35
array([ True,  True,  True, False])
>>> a
array([10, 20, 30, 40])

  在NumPy中，*用于数组间元素对应的乘法，而不是矩阵乘法，矩阵乘法可以用dot()方法来实现。

>>> A = np.array([[1,2],[3,4]])
>>> B = np.array([[0,1],[0,1]])
>>> A
array([[1, 2],
       [3, 4]])
>>> B
array([[0, 1],
       [0, 1]])
>>> A*B                    # elementwise product
array([[0, 2],
       [0, 4]])
>>> A.dot(B)               # matrix product
array([[0, 3],
       [0, 7]])
>>> np.dot(A,B)            # another matrix product
array([[0, 3],
       [0, 7]])

  有些操作，如*=，+=，-=，/=等操作，会直接改变需要操作的数组，而不是创建一个新的数组。

>>> a = np.ones((2,3), dtype = int)
>>> a
array([[1, 1, 1],
       [1, 1, 1]])
>>> b = np.random.random((2,3))
>>> b
array([[0.27020018, 0.16904478, 0.29618462],
       [0.45432616, 0.99311013, 0.56769309]])
>>> a *= 3
>>> a
array([[3, 3, 3],
       [3, 3, 3]])
>>> b += 3
>>> b
array([[3.27020018, 3.16904478, 3.29618462],
       [3.45432616, 3.99311013, 3.56769309]])
>>> b += a
>>> b
array([[6.27020018, 6.16904478, 6.29618462],
       [6.45432616, 6.99311013, 6.56769309]])
>>> a
array([[3, 3, 3],
       [3, 3, 3]])
>>> a += b              # b is not automatically converted to integer type
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Cannot cast ufunc add output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
>>> 

  当操作不同类型的数组时，最终的结果数组的类型取决于精度最宽的数组的类型。（即所谓的向上造型）

>>> a = np.ones(3, dtype=np.int32)
>>> b = np.linspace(0,pi,3)
>>> b.dtype.name
'float64'
>>> c = a+b
>>> c
array([ 1.        ,  2.57079633,  4.14159265])
>>> c.dtype.name
'float64'
>>> d = np.exp(c*1j)
>>> d
array([ 0.54030231+0.84147098j, -0.84147098+0.54030231j,
       -0.54030231-0.84147098j])
>>> d.dtype.name
'complex128'

  ndarray类实现了许多操作数组的一元方法，如求和、求最大值、求最小值等。

>>> a = np.random.random((2,3))
>>> a
array([[0.62181697, 0.26165654, 0.34994938],
       [0.95619296, 0.24614291, 0.42120462]])
>>> a.sum()
2.8569633678947346
>>> a.min()
0.24614290611891454
>>> a.max()
0.9561929625193091
>>> 

  除了上述一元方法以外，NumPy还提供了操作数组中特定行和列的一元方法，通过制定不同的axis来实现。

>>> b = np.arange(12).reshape(3,4)
>>> b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> b.sum(axis = 0)                   # sum of each column
array([12, 15, 18, 21])
>>> b.sum(axis = 1)                   # sum of each row
array([ 6, 22, 38])
>>> b.min(axis = 0)                   # min of each column
array([0, 1, 2, 3])
>>> b.min(axis = 1)                   # min of each row
array([0, 4, 8])
>>> b.max(axis = 0)                   # max of each column
array([ 8,  9, 10, 11])
>>> b.max(axis = 1)                   # max of each row
array([ 3,  7, 11])
>>> b.cumsum(axis = 1)                # cumulative sum along each row
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])
>>> b.cumsum(axis = 0)                # cumulative sum along each column
array([[ 0,  1,  2,  3],
       [ 4,  6,  8, 10],
       [12, 15, 18, 21]])
>>> 

通用方法

  NumPy提供了大量的通用数学和算术方法，比如常见的sin、cos、具体可以参考如下：

    all, any, apply_along_axis, argmax, argmin, argsort, average, bincount, ceil, clip, conj, corrcoef, cov, cross, cumprod, cumsum, diff, dot, floor, inner, inv, lexsort, max, maximum, mean, median, min, minimum, nonzero, outer, prod, re, round, sort, std, sum, trace, transpose, var, vdot, vectorize, where

>>> B = np.arange(3)
>>> B
array([0, 1, 2])
>>> np.exp(B)
array([ 1.        ,  2.71828183,  7.3890561 ])
>>> np.sqrt(B)
array([ 0.        ,  1.        ,  1.41421356])
>>> C = np.array([2., -1., 4.])
>>> np.add(B, C)
array([ 2.,  0.,  6.])

数组索引和迭代

  与Python中定义的list一样，NumPy支持一维数组的索引、切片和迭代。

>>> a = np.arange(10)**3
>>> a
array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])
>>> a[3]
27
>>> a[2:5]
array([ 8, 27, 64])
>>> a[:6:2] = -1111
>>> a
array([-1111,     1, -1111,    27, -1111,   125,   216,   343,   512,
         729])
>>> a[::-1]
array([  729,   512,   343,   216,   125, -1111,    27, -1111,     1,
       -1111])

  多维数组与一维数组相似，其在每个轴上都有一个对应的索引（index），这些索引是在一个逗号分隔的元组（tuple）中给出的。

>>> b = np.arange(15).reshape(3,5)
>>> b
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
>>> b[2,3]
13
>>> b[3,3]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: index 3 is out of bounds for axis 0 with size 3
>>> b[0,0]
0
>>> b[0,4]
4
>>> 
>>> 
>>> b[:, 1]
array([ 1,  6, 11])
>>> b[1, :]
array([5, 6, 7, 8, 9])
>>> b[-1]
array([10, 11, 12, 13, 14])
>>> b.shape
(3, 5)

  这里需要注意的是，数组的第一个索引是从0开始的。一维数组和多维数组的迭代，可以参考如下示例：

>>> for row in b:
...     print(row)
...
[0 1 2 3]
[10 11 12 13]
[20 21 22 23]
[30 31 32 33]
[40 41 42 43]

>>> for element in b.flat:
...     print(element)
...
0
1
2
3
10
11
12
13
20
21
22
23
30
31
32
33
40
41
42
43

  其中flat属性是array中的每个元素的迭代器。
shape操作
1. 改变数组的shape

  Numpy中数组shape由每个轴上元素的个数决定的。例如：

>>> import numpy as np
>>> a = np.ones((3,4), dtype = int)
>>> a
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]])
>>> a.shape
(3, 4)

  NumPy中数组的shape是可以通过多种方式进行改变的，下面展示三种改变数组shape而不改变当前数组的方法，这三种方法返回一个特定shape的数组，但是并不改变原来的数组：

>>> import numpy as np
>>> a = np.ones((3,4), dtype = int)
>>> a
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]])
>>> a.ravel()
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
>>> a
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]])
>>> b = a.ravel()
>>> b
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
>>> a
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]])
>>> c = a.reshape(2,-1)
>>> c
array([[1, 1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1, 1]])
>>> a
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]])
>>> a.T
array([[1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1]])
>>> a
array([[1, 1, 1, 1],
       [1, 1, 1, 1],
       [1, 1, 1, 1]])
>>> d = a.T
>>> d
array([[1, 1, 1],
       [1, 1, 1],
       [1, 1, 1],
       [1, 1, 1]])
>>> a.shape
(3, 4)
>>> b.shape
(12,)
>>> c.shape
(2, 6)
>>> d.shape
(4, 3)

  除此之外，NumPy还提供了可以直接修改原始数组shape的方法——resize()。resize()方法和reshape()方法的最主要区别在于，reshape()方法返回一个特定shape的数组，而resize()方法会直接更改原数组。

>>> a = np.arange(12).reshape(3,4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> a.resize(2,6)
>>> a
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11]])

2. 数组堆叠和切片

  NumPy支持将多个数据按照不同的轴进行堆叠：

>>> a = np.floor(10*np.random.random((2,2)))
>>> a
array([[0., 8.],
       [4., 8.]])
>>> b = np.floor(10*np.random.random((2,2)))
>>> b
array([[1., 4.],
       [4., 1.]])
>>> np.vstack((a,b))
array([[0., 8.],
       [4., 8.],
       [1., 4.],
       [4., 1.]])
>>> np.hstack((a,b))
array([[0., 8., 1., 4.],
       [4., 8., 4., 1.]])

  hstack()实现数组横向堆叠，vstack()实现数组纵向堆叠。

>>> from numpy import newaxis
>>> np.column_stack((a,b))
array([[4, 2],
       [2, 8]])
>>> a[:, newaxis]
array([[4],
       [2]])
>>> np.column_stack((a[:,newaxis],b[:,newaxis]))
array([[4, 2],
       [2, 8]])
>>> np.vstack((a[:,newaxis],b[:,newaxis]))
array([[4],
       [2],
       [2],
       [8]])

>>> np.r_[1:4,0,4]
array([1, 2, 3, 0, 4])

  除了支持数组的横向和纵向堆叠之外，NumPy还支持数组的横向和纵向分割，示例如下：

>>> a = np.arange(12).reshape(3,4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> np.split(a,3)
[array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
>>> np.h
np.half(         np.hanning(      np.histogram(    np.histogramdd(  np.hstack(       
np.hamming(      np.heaviside(    np.histogram2d(  np.hsplit(       np.hypot(        
>>> np.hsplit(a,4)
[array([[0],
       [4],
       [8]]), array([[1],
       [5],
       [9]]), array([[ 2],
       [ 6],
       [10]]), array([[ 3],
       [ 7],
       [11]])]
>>> np.vsplit(a,3)
[array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
>>> 

  其中，split()方法默认为横线分割。
复制和视图

  NumPy中，数组的复制有三种方式：

        Python通用的地址复制：通过 b = a 复制 a 的值，b 与 a 指向同一地址，改变 b 同时也改变 a。
        通过视图ndarray.view()仅复制值，当对 c 值进行改变会改变 a 的对应的值，而改变 c 的 shape 不改变 a 的 shape。
        ndarray.copy() 进行的完整的拷贝，产生一份完全相同的独立的复制。

>>> a = np.arange(12)
>>> a
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
>>> b = a
>>> print(a is b)
True
>>> 
>>> 
>>> c = a.view()
>>> c
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
>>> print(a is c)
False
>>> c.shape = 2,6
>>> c
array([[ 0,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11]])
>>> c[0,0] = 111
>>> c
array([[111,   1,   2,   3,   4,   5],
       [  6,   7,   8,   9,  10,  11]])
>>> a
array([111,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11])
>>> 
>>> 
>>> d = a.copy()
>>> print(a is d)
False
>>> d.shape = 2,6
>>> d
array([[111,   1,   2,   3,   4,   5],
       [  6,   7,   8,   9,  10,  11]])
>>> a
array([111,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11])
>>> d[0,0] = 999
>>> d
array([[ 999,  1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10, 11]])
>>> a
array([111,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11])
>>> 

NumPy功能和方法预览
数组创建

    arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity, linspace, logspace, mgrid, ogrid, ones, ones_like, r, zeros, zeros_like

数组转换

    ndarray.astype, atleast_1d, atleast_2d, atleast_3d, mat

操作

    array_split, column_stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack, ndarray.item, newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take, transpose, vsplit, vstack

问题

    all, any, nonzero, where

排列

    argmax, argmin, argsort, max, min, ptp, searchsorted, sort

运算

    choose, compress, cumprod, cumsum, inner, ndarray.fill, imag, prod, put, putmask, real, sum

基础统计

    cov, mean, std, var

基本线性代数

    cross, dot, outer, linalg.svd, vdot

参考：

https://segmentfault.com/a/1190000011836017
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

作者：圈圈_Master
链接：https://www.jianshu.com/p/a260a8c43e44
來源：简书
简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。
