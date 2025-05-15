# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ipython==8.32.0",
#     "ipywidgets==8.1.5",
#     "numpy==2.2.3",
#     "scipy==1.15.1",
#     "marimo",
# ]
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # EECS 16A Python Bootcamp
        ### EECS 16A: Designing Information Devices and Systems I, Fall 2024

        <!---
              Raghav Gupta (raghav.tech13@berkeley.edu)
              Ayush Pancholy (ayush.pancholy@berkeley.edu)
              Shuming Xu (smxu@berkeley.edu)
              Nikhil ograin (ncograin@berkeley.edu)
        ---->
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Table of Contents
        * [Install Test](#install-test)
        * [Overview](#overview)
        * [Assistance](#questions)
        * [Python](#python)
            * [Functions and Variables](#fnvar)
            * [Control Flow](#ctrl)
            * [List Comprehension](#lst)
        * [NumPy](#numpy)
            * [Arrays](#arrays)
            * [Slicing](#slice)
            * [Useful Functions](#funcs)
        * [Miscellaneous Functions](#misc)
        * [Questions](#qs)
        * [Feedback](#feedback)
        * [Checkoff](#checkoff)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='install-test'></a>
        # Install Test

        Welcome to your first Jupyter Notebook for 16A! Before we dive into the content of this lab, we want to make sure that all necessary packages are installed correctly. 

        Click on the block of code below (don't worry about what it does - we'll start covering that next lab) and press the Run button above to run it. Alternately, you can use `Shift + Enter` to execute and move to the next block, or `Control/Command + Enter` to execute and stay in the same block. 

        If anything strange appears (most likely a jumbled mass of text that is an error message) call over your lab TA or one of your lab assistants and they should be able to help! **<span style="color:red"> Warnings are fine, don't worry about those.</span>**

        ### **<span style="color:red">Run the code block below by clicking anywhere inside of it and pressing Shift + Enter</span>**
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # from __future__ import division
    import numpy as np
    import struct
    import time
    import warnings

    warnings.filterwarnings("ignore")
    import math
    from numpy.linalg import inv
    import scipy
    from scipy import linalg
    import random
    from IPython import display
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets
    from IPython.display import display
    import autograder
    from autograder import (
        test_q0,
        test_q1,
        test_q2,
        test_q3,
        test_q4,
        test_q5,
        test_q6,
        test_q7,
        test_q8,
        test_q9,
        test_q10,
        test_all,
    )

    print("Congratulations! Your Install is great!")
    return (
        autograder,
        display,
        fixed,
        interact,
        interact_manual,
        interactive,
        inv,
        linalg,
        math,
        np,
        random,
        scipy,
        struct,
        test_all,
        test_q0,
        test_q1,
        test_q10,
        test_q2,
        test_q3,
        test_q4,
        test_q5,
        test_q6,
        test_q7,
        test_q8,
        test_q9,
        time,
        warnings,
        widgets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='overview'></a>
        # Overview

        This mini-lab serves as an introduction to Jupyter, Python and a couple important packages we will be using throughout the semester. The lab aims to teach you proper usage of certain commands and can serve as a reference doc in the future. Even if you are a Python wizard already, we recommend that you at least look through the lab and try a few of the later problems to get re-acquainted with the functions we will be frequently using in this course.

        This lab is separated into two main parts: Guide and Questions. The Guide portion walks you through frequently used Python code, functions, and techniques. The Guide is supplemented with numerous blocks of example code to showcase concepts. The Questions portion of the lab is a collection of 10 problems meant to test your understanding of this guide. If you can answer these questions, then you have the knowledge to complete any and all programming tasks in EECS 16A.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='questions'></a>

        # Assistance
        Whenever you have a question, please raise your hand. A member of lab staff will join you shortly to help.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='questions'></a>

        # marimo notebook

        Before you get started, let's take a look at the page right now. This is what we call a marimo notebook -- an interactive interface for developing and sharing development documents. It is frequently used for data visualization projects (as you will see later in the semester) and collaborating with others.

        *Tip:* marimo notebooks comes with some handy hotkeys. You can press `tab` while typing for autocomplete and autocompletion. Also see the "Live docs" panel on the left-hand side to see documentation as you type.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='python'></a>
        # Python

        Both the labs and the homeworks in this course will require you to write some **Python** code. If you're new to programming, have no fear, as the assignments don't require more than just the fundamentals, which we will also go through during this lab; **this is not round 2 of 61A.**

        If you have any questions, feel free to refer to **[the Python Intro Guide](https://docs.google.com/document/d/1UupoCUB9cXBDffk406Z0bdYxm_9vINi45htINer55TY/edit?usp=sharing)**, which contains an overview of the language as well as features that we will be mainly using throughout the semester.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='fnvar'></a>
        ## Functions and Variables in Python

        A function is a set of statements that takes in optional inputs, runs those statements, and then returns optional data.

        Variables are containers for storing data values. These data values could be numbers, characters, words (called strings), etc. They can also point to lists or functions.
        """
    )
    return


@app.cell
def _():
    simple_number = 6
    numbers = [1, 2, 3, 4, simple_number]  # this is a list
    add = sum(
        numbers
    )  # sum is a function that takes a list of numbers and returns their sum
    print(add)
    return add, numbers, simple_number


@app.cell
def _():
    letter = "a"
    show = print  # this statement makes show point to the function print. Thus, show(param) will do the same thing as print(param)
    show(letter)
    return letter, show


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can write our own functions using the `def` keyword.""")
    return


@app.cell
def _():
    def welcome(name):
        print("Welcome to 16A " + name)


    welcome("your name here")  # type your name inside the quotes
    return (welcome,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='ctrl'></a>
        ## Control Flow in Python

        Programming languages usually contain statements that can be used to direct or "control the flow" of execution. This includes (but is not limited to) conditional statements such as `if`, `else`, and `elif`, and loop-control statements such as `while`, `for`, `break`, and `continue`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Conditional Statements: (if, else, elif)

        These check for a condition and execute a set of statements accordingly.
        """
    )
    return


@app.cell
def _():
    # Example 1: Simple if/else
    x = 16

    if x > 20:  # Asking the question, "Is x greater than 20?"
        print("if condition is True!")
    else:
        print("if condition is False!")
    return (x,)


@app.cell
def _():
    x_1 = 16
    if x_1 > 20:
        print("first if condition is True!")
    elif x_1 > 10 and x_1 < 20:
        print("first if condition is False and second if condition is True!")
    else:
        print("Neither if condition was True!")
    return (x_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Loop-Control Statements: (while, for)

        These iterate over a set of statements a specified number times or as long as a specified condition is true.
        """
    )
    return


@app.cell
def _():
    _i = 0
    while _i < 5:
        print("i:", _i)
        _i = _i + 1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Unlike while loops, which can theoretically run "forever" given the right condition, for loops serve a different purpose -- iterating a fixed number of times. For loops in Python expect an iterable object -- something that Python can iterate over, such as a list which Python can go through number by number -- to control the number of iterations. The example below is "equivalent" to the while loop in the previous example.""")
    return


@app.cell
def _():
    for _i in range(0, 5):
        print("i:", _i)
    return


@app.cell
def _():
    char_list = [1, 6, "a"]
    word = ""
    for element in char_list:
        word = word + str(element)
    print("word:", word)
    return char_list, element, word


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        All of the loop examples so far have terminated with some sort of stopping condition (ex. i < 5, i in range(0,5), element in char_list). But what if we wanted to exit a loop early? Or, what if we wanted to immediately go to the next loop iteration? These two changes can be applied using the **break** and **continue** statements, respectively.

        **break** will completely break out of the `for` or `while` loop it is in and skip any iterations it would have potentially gone through afterwards.

        **continue** will skip only the current iteration and jump to the beginning of the next one.
        """
    )
    return


@app.cell
def _():
    # Example 6: The break statement

    candies = [
        "Skittles",
        "Snickers",
        "3 Musketeers",
        "Twizzlers",
        "Kit-Kat",
        "Twix",
        "Almond Joy",
    ]
    print("Loop without break statement.")
    for candy in candies:
        print(candy)
    print("\nLoop with break statement.")  # '\n' denotes a new line
    for candy in candies:
        print(candy)
        if candy == "Kit-Kat":
            break
    return candies, candy


@app.cell
def _():
    candies_1 = [
        "Skittles",
        "Snickers",
        "3 Musketeers",
        "Twizzlers",
        "Kit-Kat",
        "Twix",
        "Almond Joy",
    ]

    print("Same Loop as above but with continue instead of break statement.")
    for candy_1 in candies_1:
        print(candy_1)
        if candy_1 == "Kit-Kat":
            continue
    print("\nLoop that skips over every-other candy.")
    for i_2 in range(len(candies_1)):
        if i_2 % 2 == 1:
            continue
        print(candies_1[i_2])
    return candies_1, candy_1, i_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In the first loop, the `continue` statement is at the end of the loop and will not skip anything, thus printing all the candies. However, in the second loop, the `continue` will potentially skip over the print statement. **Notice how the continue statement enabled us to skip over every other candy.**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='lst'></a>
        ## List Comprehension

        There are multiple ways of creating lists in Python. If you recall from the first discussion, a list is a mutable array of data. They can be created using square brackets [ ]. Elements in a list are separated by commas. Elements can be of any type (int, string, float, etc.).

        Important Python List functions: 

        - `'+'` joins lists and creates a *new* list. 

        - `len(x)` to get the length of list `x`


        Next, we will explore the idea of list comprehension, which is a compact way of creating a list from a for-loop in a single line. Please keep in mind that list comprehension is just a style suggestion; any list comprehension can be expanded to a fully fleshed out control-loop block. However, the advantage of list comprehensions is their compact yet expressive syntax.


        **For the example below, our goal is to create a list of the squares of each even integer in the range from 0 to 10 (inclusive).**
        """
    )
    return


@app.cell
def _():
    lst = []
    for i_3 in range(11):
        if i_3 % 2 == 0:
            lst = lst + [i_3**2]
    print(lst)
    return i_3, lst


@app.cell
def _():
    lst_1 = [i**2 for i in range(11) if i % 2 == 0]
    print(lst_1)
    return (lst_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The syntax for a list comprehension is as follows:<br>
        **list = <span style="color:green">[</span>**<span style="color:red">function(</span>**ITEM**<span style="color:red">)</span> **<span style="color:green">for</span>** **ITEM** **<span style="color:green">in</span>** <span style="color:orange">ITERABLE_OBJECT</span> **<span style="color:green">if</span>** <span style="color:blue">condition(</span>**ITEM**<span style="color:blue">)</span>**<span style="color:green">]</span>**

        In example 8b above:<br>

        - **ITEM** = i<br>
        - <span style="color:orange">ITERABLE_OBJECT</span> = range(11)<br>
        - <span style="color:red">function()</span> = raise **ITEM** to the second power<br>
        - <span style="color:blue">condition()</span> = is **ITEM** even?<br>

        A couple notes: 

        - list comprehensions DO NOT require a function or condition
        - list comprehensions can have nested for-loops
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='numpy'></a>
        # NumPy

        ### Pronounced NumPIE

        From the NumPy website, "NumPy is the fundamental package for scientific computing with Python. It contains among other things: a powerful N-dimensional array object." For the purposes of this course, we primarily use NumPy for its fast and fancy matrix functions. In general, Python list operations are slow; NumPy functions exploit the NumPy array object to "vectorize" the code, which usually improves the runtime of matrix calculations. **As a general rule of thumb, if a task involves vectors or matrices, you should resort to NumPy.** In addition to speeding up basic operations, NumPy contains an enormous library of matrix functions, so if you ever need to manipulate a vector or matrix, NumPy most likely already has a function implemented to suit your needs.

        **Quintessential NumPy Documentation: http://docs.scipy.org/doc/numpy/reference/index.html**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='arrays'></a>
        ## Creating a NumPy array object

        NumPy is centered around the `numpy.array()` class. This array object is extremely useful, however, it is often confused with built-in Python lists, particularly when trying to represent vectors.

        If you are not familiar with Python lists, a list is a sequence of numbers, like `1,2,3,4`, stored together. You may also see the term tuple used in notebooks and in the NumPy library. You can think of a tuple as a Python list that you cannot modify.

        A NumPy array is similar to a Python list in the sense that it also stores multiple numbers, but, as you will see later, NumPy arrays and Python Lists are NOT synonymous; you **cannot** simply apply functions to NumPy arrays as if they were Python Lists.
        """
    )
    return


@app.cell
def _(np):
    py_lst = [1, 2, 3, 4]
    np_arr = np.array(py_lst)
    print("Python list:", py_lst)
    print("NumPy array:", np_arr)
    return np_arr, py_lst


@app.cell
def _(np):
    np_arr_1 = np.empty([4, 4], dtype=np.int64)
    for i_4 in range(4):
        for j in range(4):
            np_arr_1[i_4, j] = i_4 + j
    print(np_arr_1)
    return i_4, j, np_arr_1


@app.cell
def _(np):
    np_zeros = np.zeros([5, 5])
    np_id = np.eye(5)
    print("np_zeros:\n", np_zeros)
    print("\nnp_id:\n", np_id)
    return np_id, np_zeros


@app.cell
def _(np):
    """numpy.linspace() is useful when you know the number of divisions over a certain range you want,
    i.e., you want to divide the range [0-9] into 10 equal divisions.
    """

    np_arr1 = np.linspace(0, 9, 10)
    print("np_arr1:", np_arr1)
    "numpy.arange() is useful when you know how far away each division is from one another, a.k.a. the step size.\nYou want to start at 0 and get every number that is 1 away from the previous number until you get to 9.\n"
    np_arr2 = np.arange(0, 10, 1)
    print("np_arr2:", np_arr2)
    return np_arr1, np_arr2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Note:** Dimensions in NumPy are always provided as a list/tuple. You should be able to see this in the usage of functions such as `zeros`, `empty`, `ones`, `reshape`, etc.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### NumPy array vs. Python list

        Most arithmetic operations apply to NumPy arrays in element-wise fashion. This is in contrast with arithmetic operations for Python lists, which apply via concatenation.
        """
    )
    return


@app.cell
def _(np):
    lst_2 = [1, 2, 3]
    arr = np.eye(3)
    lst2 = lst_2 + lst_2
    arr2 = arr + arr
    print("lst:", lst_2)
    print("lst + lst =", lst2)
    print("\narr:\n", arr)
    print("arr + arr =\n", arr2)
    return arr, arr2, lst2, lst_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='slice'></a>
        ## NumPy array slicing

        Array slicing is a technique in Python (and other languages) that programmers use to extract specific index-based information from an array. Array slicing answers queries such as, "What are the first/last n elements in this array?", "What are the elements in the first r rows and first c columns of this matrix?", "What is every nth element in this array?"

        **Zero indexing:** Python uses zero indexing. This means that all arrays/lists in Python start with 0 as the index of the first element. For example, in a list `lst = [a, b, c]`, the element at index 0 `lst[0]` is `a`, and the element at index 1 `lst[1]` is `b`. Similarly, `c` would be at index 2.
        """
    )
    return


@app.cell
def _(np):
    simple_arr = np.arange(0, 100, 1)
    print("\nFirst ten elements of simple_arr:", simple_arr[:10])
    print("\nLast ten elements of simple_arr:", simple_arr[-10:])
    print("\nElements 16-25 of simple_arr:", simple_arr[16:26])
    return (simple_arr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">Slicing includes the start index and excludes the end index, i.e. `simple_arr[16:26]` means to extract the values in `simple_arr` at indices in the range `[16,26)` which is the same as `[16,25]` since indices can only be integers.</span>**""")
    return


@app.cell
def _(np):
    simple_arr_1 = np.arange(0, 20, 1)
    print(
        "\nEvery-other element of simple_arr, starting from 0:", simple_arr_1[::2]
    )
    print(
        "\nEvery-third element of simple_arr, starting from 0:", simple_arr_1[::3]
    )
    print(
        "\nEvery-other element of simple_arr, starting from 10-16:",
        simple_arr_1[10:16:2],
    )
    return (simple_arr_1,)


@app.cell
def _(np):
    _i = np.array(range(25), dtype=np.int64).reshape([5, 5])
    print("i:\n", _i)
    print("\nFirst row of i:", _i[0])
    print("\nFirst column of i:", _i[:, 0])
    print("\nRows 1-3 of i:\n", _i[1:4])
    print("\nColumns 1-3 of i:\n", _i[:, 1:4])
    print("\nTop left 3x3 of i:\n", _i[:3, :3])
    print("\nEvery-other column of i:\n", _i[:, ::2])
    return


@app.cell
def _(np):
    _j = np.zeros([5, 5])
    print("j (5x5 NumPy array of all zeros):\n", _j)
    inner = np.ones([3, 3])
    print("\ninner (3x3 Numpy array of all ones):\n", inner)
    _j[1:4, 1:4] = inner
    print("\nj:\n", _j)
    print(
        "\n Notice how the values of inner are assigned to the 3x3 slice at the center of j!"
    )
    return (inner,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Slicing Summary

        Slicing a NumPy array arr follows the syntax below:<br> <br>
        arr[<span style="color:green">row_start_index</span><span style="color:blue">:</span><span style="color:red">row_end_index</span><span style="color:blue">:</span><span style="color:orange">row_step_size</span> , <span style="color:green">col_start_index</span><span style="color:blue">:</span><span style="color:red">col_end_index</span><span style="color:blue">:</span><span style="color:orange">col_step_size</span>]

        - <span style="color:green">start indices</span> are inclusive and default to 0
        - <span style="color:red">end indices</span> are exclusive and default to len(arr)
        - <span style="color:orange">step sizes</span> default to 1

        arr[<span style="color:green">start_index</span><span style="color:blue">:</span><span style="color:red">end_index</span><span style="color:blue">:</span><span style="color:orange">step_size</span>] slices rows according to the specified arguments while selecting all columns.

        arr[ : , <span style="color:green">start_index</span><span style="color:blue">:</span><span style="color:red">end_index</span><span style="color:blue">:</span><span style="color:orange">step_size</span>] slices columns according to the specified arguments while selecting all rows.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### NumPy array reshaping

        Reshaping is useful when you want to do something such as turn a vector into a matrix or vice-versa. We want to be able to do this because it is often easier to construct the desired array as a vector then reshape the vector into a matrix.
        """
    )
    return


@app.cell
def _(np):
    test_arr = np.zeros([15, 189])
    print("Shape of test_arr:", test_arr.shape)
    print("Rows in test_arr:", test_arr.shape[0])
    print("Cols in test_arr:", test_arr.shape[1])
    print("Number of elements in test_arr:", test_arr.size)
    return (test_arr,)


@app.cell
def _(np):
    test_arr_1 = np.array(range(16), dtype=np.int64)
    print("\ntest_arr:", test_arr_1)
    print("Shape of test_arr:", test_arr_1.shape)
    test_arr_4x4 = test_arr_1.reshape([4, 4])
    print("\nReshaped test_arr:\n", test_arr_4x4)
    print("Shape of test_arr_4x4:", test_arr_4x4.shape)
    test_arr_vec = test_arr_4x4.reshape(test_arr_4x4.size)
    print("\ntest_arr back as a vector:", test_arr_vec)
    print("Shape of test_arr_vec:", test_arr_vec.shape)
    return test_arr_1, test_arr_4x4, test_arr_vec


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='funcs'></a>
        ## Useful NumPy functions: (transpose(), linalg.inv(), dot(), concatenate(), vstack(), hstack(), max(), argmax())

        **Quintessential NumPy Documentation: http://docs.scipy.org/doc/numpy/reference/index.html**
        """
    )
    return


@app.cell
def _(np):
    norm = np.array(range(16), dtype=np.int64).reshape([4, 4])
    print("\nnorm:\n", norm)
    norm_transpose = np.transpose(norm)
    print("\nnorm_transpose:\n", norm_transpose)
    print("\nnorm easy transpose:\n", norm.T)
    return norm, norm_transpose


@app.cell
def _(np):
    _i = np.eye(4)
    print("\ni:\n", _i)
    i_inv = np.linalg.inv(_i)
    print("\ni_inv:\n", i_inv)
    print("\nAs expected, i == inv(i).")
    _j = np.array([[0, 1, 0, 0], [2, 0, 0, 0], [0, 0, 0, 3], [0, 0, 4, 0]])
    print("\nj:\n", _j)
    j_inv = np.linalg.inv(_j)
    print("\nj_inv:\n", j_inv)
    print(
        "\nMultiplying an invertible matrix with its inverse gives us the identity matrix.\n"
    )
    print("\nj*inv(j):\n", np.dot(_j, j_inv))
    print("\nThus, as expected, j*inv(j) == i.")
    return i_inv, j_inv


@app.cell
def _(np):
    a = np.array([[2, 3], [4, 5]])
    print("\na:\n", a)
    b = np.array([[1, 2], [0, 2]])
    print("\nb:\n", b)
    print("\nMatrix multiplication.")
    c = np.dot(a, b)
    print("a*b:\n", c)
    print("\nOrder matters in numpy.dot()!")
    d = np.dot(b, a)
    print("b*a:\n", d)
    print("Notice a*b != b*a.")
    print("\nNesting numpy.dot() to perform repeated multiplication.")
    e = np.dot(b, np.dot(b, a))
    print("b*b*a:\n", e)
    f = np.array([2, 2])
    print("\nf:", f)
    print("\nnumpy.dot() can be used to multiply an array and vector too.")
    g = np.dot(a, f)
    print("a*f:", g)
    return a, b, c, d, e, f, g


@app.cell
def _(np):
    a_1 = np.array([[2, 3], [4, 5]])
    print("\na:\n", a_1)
    b_1 = np.array([[1, 2], [0, 2]])
    print("\nb:\n", b_1)
    c_1 = np.concatenate([a_1, b_1], axis=0)
    print('\nAppend b to the "bottom" of a:\n', c_1)
    d_1 = np.concatenate([a_1, b_1], axis=1)
    print('\nAppend b to the "right" of a:\n', d_1)
    return a_1, b_1, c_1, d_1


@app.cell
def _(np):
    a_2 = np.array([[2, 3], [4, 5]])
    print("\na:\n", a_2)
    b_2 = np.array([[1, 2], [0, 2]])
    print("\nb:\n", b_2)
    c_2 = np.vstack([a_2, b_2])
    print("\nvstack a and b:\n", c_2)
    print("Notice this is equivalent to concatenate with axis=0.")
    d_2 = np.hstack([a_2, b_2])
    print("\nhstack a and b:\n", d_2)
    print("Notice this is equivalent to concatenate with axis=1.")
    return a_2, b_2, c_2, d_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='misc'></a>
        # Miscellaneous Functions
        """
    )
    return


@app.cell
def _(np):
    a_3 = 16.5
    print("a:", a_3)
    print("floor of a:", np.floor(a_3))
    print("ceiling of a:", np.ceil(a_3))
    return (a_3,)


@app.cell
def _(np):
    a_4 = np.array([0, 1, 2, 3, 16, 3, 2, 1, 0])
    print("a:", a_4)
    print("max of a =", np.max(a_4))
    print("min of a =", np.min(a_4))
    print("index of max value of a =", np.argmax(a_4))
    print("index of min value of a =", np.argmin(a_4))
    return (a_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='qs'></a>
        # Questions

        These questions are in no particular order (except for question 0, do that one first). The questions range in difficulty; some are one-liners, others require a lot more thinking. The questions are also labeled with their difficulties on a scale of 1 to 3 (with 1 being the easiest and 3 the hardest). Don't be discouraged if you hit a roadblock. Talk to your neighbors and ask for help from the lab staff.

        **<span style="color:red">You are not expected to get all of the questions correct in the first try. This is not a Python course, and the questions here are a lot harder than what you will see in future labs. This notebook is just to provide a quick introduction to Python and Juypter Notebooks and for you to have something to refer back to.</span>**

        ### Question 0

        **<span style="color:red">In order to test your code, plase run the cell below to load the autograder. There is a cell after each question that you can run in order to check your answer. The autograder is purposefully not very verbose.</span>**
        """
    )
    return


@app.cell
def _(autograder):
    autograder.test_q0()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Question 1 (Difficulty: 1)

        **<span style="color:red">Search the NumPy documentation and/or the web for a NumPy function that can solve a system of linear equations of the form `Ax=b`. Once you've found the package and function, insert into the `func` placeholder below. Often Googling `Numpy` followed by the function you'd like will give you exactly what you want. Ex: Try searching `Numpy Matrix Solver`</span>**
        """
    )
    return


app._unparsable_cell(
    r"""
    # uncomment the line below and enter the missing package/function
    func = # YOUR CODE HERE

    # Do not modify the code below
    def q1(A,b):
        return func(A,b)
    """,
    name="_"
)


@app.cell
def _(q1, test_q1, test_q1_btn):
    if test_q1_btn.value:
        test_q1(q1)
    test_q1_btn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Question 2 (Difficulty: 1)

        **<span style="color:red">Given NumPy array A, return an array that consists of every entry of A that has an even row index and an odd column index. </span>**

        See [slicing](#slice) for examples on how to do array slicing.
        """
    )
    return


@app.cell
def q2():
    def q2(A):
        """
        Input:
        A - MxN NumPy array
        Output:
        Returns an NumPy array that consists of every entry of A that has an even row index and has an odd column index.
        Example:
        A = np.array([[ 1, 2, 3, 4, 5]
                      [ 6, 7, 8, 9,10]
                      [11,12,13,14,15]
                      [16,17,18,19,20]
                      [21,22,23,24,25]])
        Output = np.array([[ 2, 4]
                           [12,14]
                           [22,24]])
        """

        # YOUR CODE HERE
    return (q2,)


@app.cell
def _(q2, test_q2, test_q2_btn):
    if test_q2_btn.value:
        test_q2(q2)
    test_q2_btn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Question 3  (Difficulty: 2)

        **<span style="color:red">Given an MxN NumPy array, first find the indices of the maximum value in each row of the array, then return the maximum index.</span>**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, question3_hint1, question3_hint2, question3_hint3):
    mo.md(f"""
    {mo.as_html(question3_hint1)}
    {mo.as_html(question3_hint2)}
    {mo.as_html(question3_hint3)}
    """)
    return


@app.cell
def q3():
    def q3(A):
        """
        Input:
        A - MxN NumPy array
        Output:
        For each row in A, determine the index corresponding to its maximum value. Then, return the greatest such index.
        Example:
        A = np.array([[0, 1, 0, 0]
                      [1, 0, 0, 0]
                      [0, 0, 0, 0]
                      [0, 0, 1, 0]])
        Output = 2
        In row 0, we have maximum value 1 at index 1.
        In row 1, we have maximum value 1 at index 0.
        In row 2, we have maximum value 0 at index 0 (since that's the first occurence of 0).
        In row 3, we have maximum value 1 at index 2.
        Thus, the output should be the maximum index, which is 2.
        """

        # YOUR CODE HERE
    return (q3,)


@app.cell
def _(q3, test_q3, test_q3_btn):
    if test_q3_btn.value:
        test_q3(q3)
    test_q3_btn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Question 4  (Difficulty: 2)

        **<span style="color:red">Given two MxN NumPy arrays, copy every-other column of array A to the right side of array B and return the new array.</span>**

        See [slicing](#slice) for examples on how to do array slicing.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, question4_hint1):
    mo.md(f"""
    {mo.as_html(question4_hint1)}
    """)
    return


@app.cell
def q4():
    def q4(A, B):
        """
        Inputs:
        A - MxN NumPy array
        B - MxP NumPy array
        Output:
        Returns an Mx(P+(N/2)) NumPy array where every-other column of A is added to the right side of B in order,
        starting from index 0.
        Example:
        A = np.array([[1,0,0,2]
                      [0,1,0,2]
                      [0,0,1,2]])
        B = np.array([[1,2,3]
                      [4,5,6]
                      [7,8,9]])
        Output = np.array([[1,2,3,1,0]
                           [4,5,6,0,0]
                           [7,8,9,0,1]])
        """

        # YOUR CODE HERE
    return (q4,)


@app.cell
def _(q4, test_q4, test_q4_btn):
    if test_q4_btn.value:
        test_q4(q4)
    test_q4_btn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Question 5  (Difficulty: 2)

        **<span style="color:red">For any given N, u = [1,2,3,...,N] and v = [2017,2018,2019,...,2017+N-1]. Write a function that returns a vector that contains the following sequence: [1`*`2017, 2`*`2018, 3`*`2019,...,N`*`(2017+N-1)]. </span>**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, question5_hint1):
    mo.md(f"""
    {mo.as_html(question5_hint1)}
    """)
    return


@app.cell
def q5():
    def q5(N):
        """
        Input:
        N - number of elements in u and v.
        Output:
        Returns the sequence: np.array([1*2017,2*2018,...,N*(2017+N-1)])
        Example:
        N = 5
        Output = np.array([2017, 4036, 6057, 8080, 10105])
        """

        # YOUR CODE HERE
    return (q5,)


@app.cell
def _(q5, test_q5, test_q5_btn):
    if test_q5_btn.value:
        test_q5(q5)
    test_q5_btn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Question 6  (Difficulty: 1)

        **<span style="color:red">Given a NumPy vector v, shift all of the elements in v by n steps to the right; values that "fall off" the right end of v get inserted at the beginning of v, thus the length of v is preserved. You can either attempt to implement this on your own, or, (hint hint) try searching for a related NumPy function that does some/all of the work for you... </span>**
        """
    )
    return


@app.cell
def q6():
    def q6(v, N=10):
        """
        Input:
        v = NumPy vector
        N = number of steps to shift v to the right
        Output:
        Returns v shifted to the right by N steps.
        Example:
        v = np.array([0,1,2,3,4,5])
        N = 3
        Output = np.array([3,4,5,0,1,2])
        """

        # YOUR CODE HERE
    return (q6,)


@app.cell
def _(q6, test_q6, test_q6_btn):
    if test_q6_btn.value:
        test_q6(q6)
    test_q6_btn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Question 7  (Difficulty: 2)

        **<span style="color:red">Given an MxM identity matrix, convert this to an (M-N)x(M-N) identity matrix WITHOUT using numpy.eye().</span>**
        """
    )
    return


@app.cell
def _(np):
    def q7(I=np.eye(10), N=4):
        """
        Input:
        I - MxM NumPy array representing the identity matrix
        N - number of rows and columns to cut from I
        Output:
        Returns an (M-N)x(M-N) NumPy identity array.
        Example:
        I = np.eye(10)
        N = 8
        Output = np.array([[1,0]

                           [0,1]])
        """
    return (q7,)


@app.cell
def _(q7, test_q7, test_q7_btn):
    if test_q7_btn.value:
        test_q7(q7)
    test_q7_btn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Question 8  (Difficulty: 3)

        **<span style="color:red">Given a square NxN NumPy array A, return a Python list of the values along the diagonal of A, sorted in descending order.</span>**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, question8_hint1, question8_hint2):
    mo.md(f"""
    {mo.as_html(question8_hint1)}
    {mo.as_html(question8_hint2)}
    """)
    return


@app.cell
def q8():
    def q8(A):
        """
        Input:
        A - NxN NumPy array
        Output:
        Returns a Python list containing the diagonal of A sorted in descending order.
        Example:
        A = np.array([[1,2,3]
                      [4,5,6]
                      [7,8,9]])
        Output = [9,5,1]
        """

        # YOUR CODE HERE
    return (q8,)


@app.cell
def _(q8, test_q8, test_q8_btn):
    if test_q8_btn.value:
        test_q8(q8)
    test_q8_btn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Question 9 (Difficulty: 3)

        **<span style="color:red">Given an MxN matrix, A, and an NxM matrix, B, concatenate (side-by-side) the first p rows of A with the transpose of the last p columns of B.</span>**
        """
    )
    return


@app.cell
def q9():
    def q9(A, B, p):
        """
        Input:
        A - MxN NumPy array
        B - NxM NumPy array
        p - number of rows from A to concatenate with number of columns from B
        Output:
        Returns the side-by-side concatenation of the first p rows of A with the transpose of the last p columns of B.
        Example:
        A = np.array([[1,1,1]
                      [1,1,1]
                      [1,1,1]
                      [1,1,1]])
        B = np.array([[1,2,3,4]
                      [5,6,7,8]
                      [9,10,11,12]])
        p = 2
        Output = np.array([[1,1,1,3,7,11]
                           [1,1,1,4,8,12]])
        """

        # YOUR CODE HERE
    return (q9,)


@app.cell
def _(q9, test_q9, test_q9_btn):
    if test_q9_btn.value:
        test_q9(q9)
    test_q9_btn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Question 10 (Difficulty: 3)

        **<span style="color:red">Given two differently sized matrices, "pad" the matrices with the smaller dimensions with rows/columns of zeros until they are the same size as one another. Add the padding to the bottom (if adding rows) and to the right (if adding columns). </span>**

        See [slicing](#slice) for examples on how to do array slicing. Recall that you can assign to a slice.

        This is typically the hardest question for students.

        Note: You can return multiple values with commas! For example, use `return A, B` to return both A and B.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, question10_hint1):
    mo.md(f"""
    {mo.as_html(question10_hint1)}
    """)
    return


@app.cell
def q10():
    def q10(A, B):
        """
        Input:
        A - MxN NumPy array
        B - YxZ NumPy array
        Output:
        Returns the zero-padded versions of each array such that they are of equivalent dimensions.
        Padding is added to the bottom and right.
        Example:
        A = np.array([[1,2,3]
                      [4,5,6]])
        B = np.array([[1,1]
                      [1,1]
                      [1,1]])
        Output = np.array([[1,2,3]
                           [4,5,6]
                           [0,0,0]]),
                  np.array([[1,1,0]
                            [1,1,0]
                            [1,1,0]])
        """

        # YOUR CODE HERE
    return (q10,)


@app.cell
def _(q10, test_q10, test_q10_btn):
    if test_q10_btn.value:
        test_q10(q10)
    test_q10_btn
    return


@app.cell
def _(q1, q10, q2, q3, q4, q5, q6, q7, q8, q9, test_all, test_all_btn):
    if test_all_btn.value:
        test_all(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10)
    test_all_btn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='feedback'></a>
        ## Feedback
        If you have any feedback to give the teaching staff about the course (lab content, staff, etc), you can submit it through this Google form. Responses are **fully anonymous** and responses are actively monitored to improve the labs and course. Completing this form is **not required**.

        [Anyonymous feedback Google form](https://docs.google.com/forms/d/e/1FAIpQLSdSbJHYZpZqcIKYTw8CfpfrX6OYaGzqlgBtKfsNKEOs4BzZJg/viewform?usp=sf_link)

        *If you have a personal matter to discuss or need a response to your feedback, please contact <a href="mailto:eecs16a.lab@berkeley.edu">eecs16a.lab@berkeley.edu</a> and/or <a href="mailto:eecs16a@berkeley.edu">eecs16a@berkeley.edu</a>*.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='checkoff'></a>
        ## Checkoff
        To receive credit, all labs will require the submission of a checkoff Google form. This link will be at the bottom of each lab. Both partners should fill out the form (you should have one submission per person), and feel free to use the same Google account/computer to fill it out as long as you have the correct names and student IDs.

        [Fill out the checkoff Google form.](https://docs.google.com/forms/d/e/1FAIpQLSfIOjvEJXew-M0-h9uJ3C25UOdmmABFK0GGNl3o9p7po7Cc0A/viewform?usp=sf_link)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Marimo Instructions [Do Not Change]""")
    return


@app.cell
def _(mo):
    test_q1_btn = mo.ui.run_button(label="Test Q1")
    test_q2_btn = mo.ui.run_button(label="Test Q2")
    test_q3_btn = mo.ui.run_button(label="Test Q3")
    test_q4_btn = mo.ui.run_button(label="Test Q4")
    test_q5_btn = mo.ui.run_button(label="Test Q5")
    test_q6_btn = mo.ui.run_button(label="Test Q6")
    test_q7_btn = mo.ui.run_button(label="Test Q7")
    test_q8_btn = mo.ui.run_button(label="Test Q8")
    test_q9_btn = mo.ui.run_button(label="Test Q9")
    test_q10_btn = mo.ui.run_button(label="Test Q10")
    test_all_btn = mo.ui.run_button(label="Test All")
    return (
        test_all_btn,
        test_q10_btn,
        test_q1_btn,
        test_q2_btn,
        test_q3_btn,
        test_q4_btn,
        test_q5_btn,
        test_q6_btn,
        test_q7_btn,
        test_q8_btn,
        test_q9_btn,
    )


@app.cell
def _(mo):
    question3_hint1 = mo.accordion( {"Hint 1": "There is a function to find the index of the maximum value in a vector."})
    question3_hint2 = mo.accordion( {"Hint 2": "If you write `for x in matrix: ...`, what shape is `x`? "})
    question3_hint3 = mo.accordion( {"Hint 3": "[List Comprehensions](#lst) might be useful."})
    return question3_hint1, question3_hint2, question3_hint3


@app.cell
def _(mo):
    question4_hint1 = mo.accordion( {"Hint": "Are there any [useful functions](#funcs) you can use to horizontally stack arrays?"})
    return (question4_hint1,)


@app.cell
def _(mo):
    question5_hint1 = mo.accordion( {"Hint": "You might want to create vectors u and v."})
    return (question5_hint1,)


@app.cell
def _(mo):
    question8_hint1 = mo.accordion({"Hint 1": "NumPy has a function to return the elements along a diagonal. Try searching for it!"})
    question8_hint2 = mo.accordion({"Hint 2": "Python's `sorted` function can operate on lists, but not NumPy arrays. `sort` is another similar but subtly different function. What type of object does the function from the first hint return?"})
    return question8_hint1, question8_hint2


@app.cell
def _(mo):
    question10_hint1 = mo.accordion({"Hint": "There might be a NumPy function that does something similar/exactly to this, but it's good practice to try this yourself. Consider making temporary matrices of all zeros with a specific size."})
    return (question10_hint1,)


if __name__ == "__main__":
    app.run()
