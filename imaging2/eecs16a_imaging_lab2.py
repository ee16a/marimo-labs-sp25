# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.0",
#     "numpy==2.2.3",
#     "pyserial==3.5",
#     "tqdm==4.67.1",
# ]
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Imaging Lab 2: Single Pixel Scanning

        ### EECS 16A: Designing Information Devices and Systems I, Fall 2024
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # Table of Contents

        * [Instructions](#instructions)
        * [Lab Policies](#policies)
        * [Overview](#overview)
        * [Task 1: Images, Vectors, and Matrices](#images)
            * [Task 1a: Working with Images](#task2a)
            * [Task 1b: Scanning Mask Matrix](#task2b)
        * [Task 2: Imaging Real Pictures](#task3)
        * [Feedback](#feedback)
        * [Checkoff](#checkoff)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        <a id='instructions'></a>
        ## Instructions

        * Complete this lab by filling in all of the required sections, marked with `"YOUR CODE HERE"` or `"YOUR COMMENTS HERE"`.
        * When you finish, submit a checkoff request to get checked off (i.e. earn credit) for this lab. Be ready to answer a few questions to show your understanding of **each section**.
        * Labs will be graded based on completion for **teams of 2 students**.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='policies'></a>
        ## Lab Policies
        * **YOU MUST ATTEND THE LAB SECTION YOU ARE ENROLLED IN. If you anticipate missing a section, please notify your GSI in advance.**
        * **You are required to return all parts checked out at the beginning of the lab section unless told otherwise.**
        * **You are free to stay for the full allotted time and hack around with the lab equipment, but please reserve the GSI's time for lab-related questions.**
        * **Food and drinks are not allowed in the lab.** 
        * **Clean up, turn off all equipment, and log off of computers before leaving.**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # <a id='overview'><span style='color:blue'>Overview</span></a>
        <center>
        <img src="./public/systemdiagram.png" style="height:256px" />
        </center>

        This week, you will photograph a real-life object pixel-by-pixel by writing code in your Marimo notebook to display the captured image. 

        First, you will write code to generate the "mask" patterns that the projector uses to scan the object. 

        Then, you will write code to recreate the image from light sensor readings provided.

        The single pixel imaging process (including some circuit detail, which you do not need to know) would involve the following steps:
        - The projector illuminates the object with a mask.
        - The ambient light sensor detects the total amount of light reflected off the object. More light leads to more current through the sensor.
        - The analog circuit converts the sensor's current into an output voltage. More light $\rightarrow$ higher sensor current $\rightarrow$ higher output voltage.
        - This analog voltage is converted into a digital brightness value.   

        <b>Note:</b> In the real world, we come across random irregular fluctuations while taking measurements. This is called noise. It is important to consider noise while designing any system, and this lab is no different. You will learn more about noisy imaging in the Imaging 3 lab.
        </font>
        """
    )
    return


@app.cell
def _():
    # Import necessary libraries

    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='images'></a>
        # <span style='color:blue'>Task 1: Images, Vectors, and Matrices </span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='task2a'></a>
        ## <span style="color:blue">Task 1a: Working with Images</span>

        <br>

        How can we represent an image? Consider a 5x5 grayscale image, where each of the 25 pixel intensities vary in shades of gray. One way to represent this is with a 2D matrix (2D NumPy array). The values stored in this array, varying from 0 to 1, correspond to different shades of gray: the lower the pixel value, the darker the pixel, with 0 being completely black and 1 being completely white. 

        For example, take the 5x5 **`gradient_image`** shown below. Starting from the top-left pixel (pixel[0,0]), each pixel becomes progressively brighter as you traverse the image row-by-row. 

        Note: We will be using 0 indexing in lab as most programming languages (including Python) index in lists starting from 0.

        <center>
        <img src="public/gradient.JPG" align="center" style="height:200px" />
        <figcaption>Gradient image example</figcaption>
        </center>

        We can create this in Python using a $5 \times 5$ NumPy 2D array called **`gradient_image`** with *linearly-spaced* floating point values from 0 to 1. The Python code to generate this is provided for you below. Take a look at the numerical 2D array and the corresponding image that is displayed by using the `imshow` function.
        </font>
        """
    )
    return


@app.cell
def _(np, plt):
    # A 5x5 gradient image with values from 0 to 1.

    gradient_image = np.linspace(0, 1, 25).reshape([5, 5])
    print(gradient_image)
    plt.imshow(gradient_image, cmap = "gray", interpolation = "nearest")
    return (gradient_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">What color does 1.0 correspond to? What about 0?</span>**""")
    return


@app.cell(hide_code=True)
def _(answer_1, mo):
    mo.md(f"""
    {mo.as_html(answer_1)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Instead of treating our images as 2D matrices, we can "reshape" or "flatten" our images into 1-D vectors. That is, instead of having a $5 \times 5$ matrix for our image, we can represent it using a $25 \times 1$ vector. This makes it simpler for us to use the linear algebra techniques learned in class for image processing. 

        Let's look at the 3x3 example image below (colored for illustrative purposes). How can we transform this 2D vector matrix into a 1-D column vector?   

        Essentially, the $0$th row is transposed (or flipped on its side by rotating 90 degress clockwise), such that its left-most element is on top and its right-most element is on the bottom. The $1$st row is also transposed on its side in the same way and appended below. These steps are repeated for each subsequent row of the original 2D image until you build a $9 \times 1$ **column vector**.    

        <center>
        <img src="public/matrix_to_col_new.png" style="width:500px"/>
        </center>

        Mathematically, each pixel value in the $3 \times 3$ image is represented as a variable $p_{ij}$, where $i$ is the row and $j$ is the column associated with the pixel location. This same image represented as a 1-D column vector (called $\vec{i}$) is:

        $$\vec{i} = \begin{bmatrix} p_{00} \\ p_{01} \\ p_{02} \\ p_{10} \\ p_{11} \\ p_{12} \\ p_{20} \\ p_{21} \\ p_{22} \end{bmatrix}$$    

        The procedure described above can be used to convert any $N \times M$ 2D image into a `num_pixels` $\times 1$ **column vector**, where `num_pixels` $= N \times M$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">Convert the 5x5 `gradient_image` that you created above into a 25x1 column vector `gradient_image_vector` and display it. You will find the command `np.reshape` helpful. What pattern do you notice? Think about why you see this pattern.</span>**""")
    return


@app.cell(hide_code=True)
def _(answer_2, mo):
    mo.md(f"""
    {mo.as_html(answer_2)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='task2b'></a>
        ## <span style="color:blue">Task 1b: Scanning Mask Matrix</span>

        Next, we will create a "mask" matrix (array) to enable our projector to illuminate and scan individual pixels, one at a time. This is the magic behind our single pixel camera. 

        If **`gradient_image_vector`** is represented by the column vector variable $\vec{i}$, the act of transforming $\vec{i}$ by a matrix $H$ into another 1D column vector $\vec{s}$ is represented mathematically as:

        $$\vec{s} = H \vec{i}$$

        This matrix-vector multiplication represents what happens when we scan an image with our single pixel camera! In the context of a real-world imaging system, $H$ represents the scanning "mask matrix," whose rows are projected one-by-one onto the image we want to scan. $\vec{s}$ represents digitized readings from the analog circuit's light sensor. 

        Each element $s_k$ of $\vec{s}$ corresponds to one scan (using one row $k$ of $H$, that we refer to as $H_k$). Each 1D **row of $H$** represents a **mask**. But what is a mask? Here, a mask is a way to highlight certain locations in the image while hiding others during scanning. For a 3x4 image (where 3 = height, 4 = width), a mask taken from **row 0 of $H$** is represented as the $1 \times 12$ row vector below: 

        $$
        H_0 
        = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\end{bmatrix}
        $$

        However, the mask must first be converted into its 2D form, as shown below, before it's projected over the 2D image. The mask exposes only the top-left pixel of the 2D image and hides all other pixels. Note that you can convert a 2D mask into a 1D row of $H$ by appending each of the 2D mask's rows to the right of the previous row.
        <br><br>
        <center>
        <img src="public/black_hite.png" style="width:400px"/>
        </center>

        To expose each pixel of the 3x4 image $\vec{i}$ individually, we would need a 12x12 $H$ that has 12 masks (rows), each with a single white "exposed" pixel in a unique location. This means that **row 1 of $H$** (exposing $iv_{01}$) would look like:

        $$
        H_1 
        = \begin{bmatrix} 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\end{bmatrix}
        $$

        <br><br>
        <center>
        <img src="public/black_white_shifted.jpg" style="width:400px"/>
        </center>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The process of masking an image, then uncovering one pixel at a time, and sensing the resultant ambient light performs the matrix multiplication $\vec{s} = H \vec{i}$ in real life. This equation implies that each element of the sensor output vector $\vec{s}$ can be determined as:

        $$s_k = H_k \vec{i}$$

        Where the $k$th sensor reading is determined by the mask given by $k$th row of $H$, $H_k$. Thus, projecting the 2D representation of $H_0$ shown above onto a 3x4 image represented by the column vector $\vec{i}$ to obtain the sensor reading $s_0$ would be mathematically equivalent to:

        $$
        s_0 = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\end{bmatrix} \vec{i}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">What dimensions does the mask matrix $H$ have for a 5x5 image? Why? </span>**""")
    return


@app.cell(hide_code=True)
def _(answer_3, mo):
    mo.md(f"""
    {mo.as_html(answer_3)}
    """)
    return


@app.cell(hide_code=True)
def _(hint_1, mo):
    mo.md(
        f"""
        **<span style="color:red">
        Create the mask matrix $H$ for a 5x5 image.</span>**
        {mo.as_html(hint_1)}
        """
    )
    return


app._unparsable_cell(
    r"""
    # TODO: Create the mask matrix `H` for scanning a 5x5 image (be careful about the dimensions!)
    H = # YOUR CODE HERE

    # Test H for correctness
    test1b_H(H)

    # Display this matrix
    plt.imshow(H, cmap = \"gray\", interpolation = \"nearest\")
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **<span style="color:red">
        Multiply the $H$ matrix with `gradient_image_vector`. Remember to use `np.dot` to do matrix multiplication and pay attention to the order of multiplcation!</span>**
        """
    )
    return


app._unparsable_cell(
    r"""
    # TODO: Multiply `H` and `gradient_image_vector`
    s = # YOUR CODE HERE

    # Display the result and compare it to `gradient_image_vector`
    plt.imshow(s, cmap = \"gray\", interpolation = \"nearest\")
    plt.xticks([])
    plt.yticks(np.arange(0, 30, 5))
    plt.show()
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">Is the resultant `s` equal to `gradient_image_vector`? Why?</span>**""")
    return


@app.cell(hide_code=True)
def _(answer_4, mo):
    mo.md(f"""
    {mo.as_html(answer_4)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        What happens when this matrix multiplication is performed? To reiterate, each row of $H$ is responsible for "illuminating," or selecting, a single pixel in the gradient image. `gradient_image_vector` was created by converting the 5x5 `gradient_image` into a 1D *column vector*. Similarly, *every row* in $H$ can be represented as a 5x5 image that, in real imaging, would be projected over `gradient_image`. 

        **<span style="color:red">
        Iterate through each row of the matrix $H$. *Reshape* each row into a 5x5 image, and check that each row illuminates a unique pixel of the original 5x5 image! Based on $\vec{s} = H \vec{i}$, why are the rows of $H$ used for masking when $\vec{i}$ is a column vector?</span>**
        """
    )
    return


@app.cell(hide_code=True)
def _(answer_5, mo):
    mo.md(f"""
    {mo.as_html(answer_5)}
    """)
    return


app._unparsable_cell(
    r"""
    # Iterate through rows of matrix H and form individual masks
    plt.figure(figsize = (20, 20)) 
    for k in range(25):
        plt.subplot(5, 5, k + 1)

        mask = # YOUR CODE HERE

        plt.imshow(mask, cmap = \"gray\", interpolation = \"nearest\")
        plt.title(\"Mask \" + str(k) + \" = Row \" + str(k) + \" of Matrix H\")
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Each of the images above are masks. During a single scan, we project one of these masks over our object. The white pixel illuminates a particular location on the object that we want to capture, and the black pixels obscure the other parts of the object. Thus, using the rows of $H$, we gather information one pixel at a time.

        Let's try to make another mask matrix, $H_{alt}$, that's a little more complicated. We want $\vec{s}$ to contain information on each pixel of the image, but in a random order. Sensing pixels in a random order and being able to reconstruct the right image is a good way to test the correctness of our imaging model. 

        **<span style="color:red">
        Generate $H_{alt}$ for a 5x5 image that illuminates each pixel of the image one at a time, but in a random order. Multiply $H_{alt}$ by `gradient_image_vector` to produce the new output vector $\vec{s}_{alt}$.
        </span>**
        <br><br>
        <center>
        <img src="public/H_alt_new_4x4.png" style="width:300px"/>
            <figcaption>A variation of $H_{alt}$ for a 4x4 image. </figcaption>
        </center>
        """
    )
    return


@app.cell(hide_code=True)
def _(hint_2, hint_3, mo):
    mo.md(f"""
        {mo.as_html(hint_2)}
        {mo.as_html(hint_3)}
    """)
    return


app._unparsable_cell(
    r"""
    # TODO: Create the new mask matrix `H_alt` for a 5x5 image
    H_alt = # YOUR CODE HERE

    # Test H_alt for correctness
    test1b_H_alt(H_alt)

    # Display `H_alt`
    plt.figure()
    plt.imshow(H_alt, cmap = \"gray\", interpolation = \"nearest\")

    # TODO: Multiply `H_alt` and `gradient_image_vector`
    s_alt = # YOUR CODE HERE

    # Display the result `s` and compare to `gradient_image_vector`
    plt.figure()
    plt.imshow(s_alt, cmap = \"gray\", interpolation = \"nearest\")
    plt.xticks([])
    plt.yticks(np.arange(0, 30, 5))
    plt.show()
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Because of how we designed `H_alt`, `s_alt` is clearly different from `gradient_image_vector`. Each pixel of `gradient_image_vector` is still illuminated only once by `H_alt`, but the order in which the pixels are illuminated has changed. Therefore, we say that `s_alt` is a "scrambled" version of `gradient_image_vector`. How could we "reconstruct" $\vec{s}_{alt}$ back into the original `gradient_image_vector`? 

        Recall that our original matrix $H$ was actually the **identity matrix** $I_n$. In other words, the original $\vec{s}$ was computed as:

        $$ \vec{s} = H \vec{i} = I_n \vec{i}$$

        <br />    
        Using the alternate mask, $H_{alt}$, we compute the alternate output $\vec{s}_{alt}$ as:

        $$ \vec{s}_{alt} = H_{alt} \vec{i} $$

        To "reconstruct" $\vec{s}_{alt}$ back into the original `gradient_image_vector` (i.e. $\vec{i}$), we must find a matrix $M$ that multiplies $\vec{s}_{alt}$ to make the following true:

        $$ M \vec{s}_{alt} = \vec{i} $$

        i.e.

        $$ M H_{alt} \vec{i} = \vec{i} $$

        **<span style="color:red">What should M be to recover $\vec{i}$?</span>**
        """
    )
    return


@app.cell(hide_code=True)
def _(answer_6, mo):
    mo.md(f"""
    {mo.as_html(answer_6)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">Write code to reconstruct `gradient_image_vector` from `s_alt`.</span>**""")
    return


app._unparsable_cell(
    r"""
    # TODO: Reconstruct `gradient_image_vector`
    M = # YOUR CODE HERE
    gradient_image_vector_reconstruct = # YOUR CODE HERE

    # Display M
    plt.figure()
    plt.imshow(M, cmap = \"gray\", interpolation = \"nearest\")
    plt.title(\"M\")
    plt.show()

    # Display M*H_alt
    plt.figure()
    plt.imshow(np.dot(M,H_alt), cmap = \"gray\", interpolation = \"nearest\")
    plt.title(\"M*H_alt\")
    plt.show()

    # Display the result
    plt.imshow(gradient_image_vector_reconstruct, cmap = \"gray\", interpolation = \"nearest\")
    plt.xticks([])
    plt.yticks(np.arange(0, 30, 5))
    plt.show()
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='task3'></a>
        ## <span style="color:blue">Task 2: Imaging Real Pictures</span>

        Finally, we will use our two matrices to image a real object; any drawing of your choice!

        There are index cards and markers at the GSI desk; **<span style="color:red">take an index card and draw something on its blank (no lines) side.** Feel free to be creative and draw something cool! The course staff always loves to see cool drawings! Keep in mind though that this is not the most sophisticated imaging setup ever. Keep details to a minimum and use large features where possible to ensure a good quality image.

        Because our object is fairly large, we want each individual mask to have dimensions 30x40 (i.e. height = 30, width = 40) to match the 4:3 (W:H) aspect ratio of the index card. Think about how big the mask matrix was for the 5x5 example. How big must it be for a 30x40 picture?

        **<span style="color:red">
        Recreate both the $H$ and $H_{alt}$ masks to match these new dimensions. </span>**
        """
    )
    return


app._unparsable_cell(
    r"""
    # TODO: Recreate `H`
    H = # YOUR CODE HERE

    plt.figure(figsize = (10, 10))
    plt.imshow(H, cmap = 'gray', interpolation=\"nearest\")
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    # TODO: Recreate `H_alt`      
    H_alt = # YOUR CODE HERE

    plt.figure(figsize = (10, 10))
    plt.imshow(H_alt, cmap = 'gray', interpolation=\"nearest\")
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's make sure that the two matrices we made are invertible and the correct size. Run the code block below to invert the matrices and test dimensions - if any of the lines fail, it means the code used to generate either matrix resulted in a incorrect size or non-invertible, linearly dependent matrix, which is incorrect.""")
    return


@app.cell
def _(H, H_alt, test_masks_img2):
    test_masks_img2(H, H_alt)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Our mask matrices must be saved as files before they can be used to perform real imaging. The files are read by our imaging script, as seen below.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">Run the cell below to save `H` and `H_alt`!</span>**""")
    return


@app.cell
def _(np):
    np.save('H_alt.npy', np.array([]))
    return


@app.cell
def _(H, H_alt, np):
    np.save('H.npy', H)
    np.save('H_alt.npy', H_alt)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        After the sensor readings have been captured, load the sensor reading vector into the cell below. Once again, here is the equation relating $H$, the sensor reading column vector $\vec{s}$, and the image column vector $\vec{i}$:

        $$\vec{s} = H \vec{i}$$

        **<span style="color:red">Recreate the image from the sensor readings obtained with `H`.</span>**
        """
    )
    return


app._unparsable_cell(
    r"""
    # TODO: Create the image vector from `H` and `sr`
    # Hint: `H` is a special matrix. What is so special about this matrix?
    iv = # YOUR CODE HERE

    img = np.reshape(iv, (30, 40))
    plt.figure(figsize = (8, 8))
    plt.imshow(img, cmap = 'gray', interpolation=\"nearest\")
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **<span style="color:red">
        Does your recreated image match the real image? What are some problems you notice? 
        </span>**

        Here is an example of a picture we took using this setup:

        <center>
        <img src="public/ee16a_picture.png"/>
        </center>
        """
    )
    return


@app.cell(hide_code=True)
def _(answer_7, mo):
    mo.md(f"""
    {mo.as_html(answer_7)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Observe your sensor reading. Since we're scanning pixels of the image in a random order, it should be a scrambled version of the image.""")
    return


app._unparsable_cell(
    r"""
    # TODO: Create the image vector from `H_alt` and `sr`
    # Hint: You need to perform a matrix operation before multiplying
    iv = # YOUR CODE HERE 

    img = np.reshape(iv, (30, 40))
    plt.figure(figsize = (8, 8))
    plt.imshow(img, cmap = 'gray', interpolation=\"nearest\")
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **<span style="color:red">The sensor reading is a scrambled version of the image. Were you able to reconstruct the image correctly? How did it get "unscrambled"?  </span>**

        `YOUR ANSWER HERE`
        """
    )
    return


@app.cell(hide_code=True)
def _(answer_8, mo):
    mo.md(f"""
    {mo.as_html(answer_8)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You are done for the week! Save your code and setup for next lab, where you will illuminate multiple pixels per mask!""")
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
        When you are ready to get checked off, fill out the checkoff google form. **[Checkoff Form](https://docs.google.com/forms/d/e/1FAIpQLSfIOjvEJXew-M0-h9uJ3C25UOdmmABFK0GGNl3o9p7po7Cc0A/viewform?usp=sf_link)**

        Your GSI or a Lab Assistant will join you when they are available and go through some checkoff questions with your group. They will go through the checkoff list in order. Please be patient!

        #### Post-checkoff Clean Up: (this applies to each week's lab)
        2. Throw away any trash at your station
        4. SIGN OUT of the computers - DO NOT SHUT DOWN
        5. Check that the projector is powered off and disconnected
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#Marimo UI Elements (do not edit)""")
    return


@app.cell
def _(mo):
    hint_1 = mo.accordion({"Hint": "Google the function `np.eye`."})
    hint_2 = mo.accordion({"Hint": "We can use rows from the existing H matrix and then shuffle their order randomly. Consider using np.random.permutation() for this. The code to correctly generate H_alt should only require you to type 1 short line."})
    hint_3 = mo.accordion({"Hint 2": "Here's one of many variations of H_alt for an image of size 4x4."})
    hint_4 = mo.accordion({"Hint": "Because `HSingle` is a special matrix, technically you do not need to perform any matrix operations."})
    hint_5 = mo.accordion({"Hint": "You need to perform a matrix operation before multiplying"})
    return hint_1, hint_2, hint_3, hint_4, hint_5


@app.cell
def _(mo):
    answer_1 = mo.ui.text_area(placeholder="Type your answer here …")
    answer_2 = mo.ui.text_area(placeholder="Type your answer here …")
    answer_3 = mo.ui.text_area(placeholder="Type your answer here …")
    answer_4 = mo.ui.text_area(placeholder="Type your answer here …")
    answer_5 = mo.ui.text_area(placeholder="Type your answer here …")
    answer_6 = mo.ui.text_area(placeholder="Type your answer here …")
    answer_7 = mo.ui.text_area(placeholder="Type your answer here …")
    answer_8 = mo.ui.text_area(placeholder="Type your answer here …")
    return (
        answer_1,
        answer_2,
        answer_3,
        answer_4,
        answer_5,
        answer_6,
        answer_7,
        answer_8,
    )


if __name__ == "__main__":
    app.run()
