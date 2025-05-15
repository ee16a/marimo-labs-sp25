import marimo

__generated_with = "0.11.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Imaging Lab 3: Multipixel Scanning

        ## EECS 16A: Foundations of Signals, Dynamical Systems, and Information Processing, Spring 2025

        <!--- 
            Raghav Gupta raghavgupta@berkeley.edu 
            Nikhil Ograin ncograin@berkeley.edu
        --->
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Table of Contents

        * [Instructions](#instructions)
        * [Lab Policies](#policies)
        * [Overview](#overview)
        * [Task 1: Generating Multipixel Scanning Matrices](#matrixGenIntro)
            * [Task 1a: Imaging Mask Matrix Practice](#simpleMatrixGen)
            * [Task 1b: Generating a Random Binary Mask Matrix](#randomBinaryMatrixGen)
        * [Task 2: Imaging Simulator](#simulatorIntro)
            * [Task 2a: Constructing an Ideal Sensor Model](#idealSensor)
                * [Image Reconstruction Using the Ideal Sensor Model + Matrix Inverse](#idealReconstruction)
            * [Task 2b: Handling System Non-Idealities](#nonidealities)
                * [Noise *(Why So Grainy? ☹)*](#noiseSimulation)
            * [Task 2c: Eigenanalysis & the Robustness of Inverse-Based Reconstruction](#eigenanalysis)
                * [Graphical Interpretation](#graphicalInterpretation)
                * [Revisiting the Identity Matrix](#revisitingIdentity)
                * [Comparing Scanning Matrices](#comparingScanning)
        * [Task 3: Scanning Images](#scanningImages)
            * [Task 3a: Single Pixel Sanity Check](#singlePixel)
            * [Task 3b: Real Multipixel Imaging](#realImaging)
        * [Task 4: Understanding Multipixel Use-Cases](#useCases)
        * [Feedback](#feedback)
        * [Checkoff](#checkoff)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
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
        <a id='overview'></a>
        # <span style='color:blue'>Overview</span>

        Recall that in the last lab, you illuminated the object pixel-by-pixel. This week, you'll flex your linear algebra skills and try something different. You will experiment with imaging methods that illuminate *multiple pixels* at a time. You'll find that if we design our masks in a clever way, our imaging system can be much more robust to noise than the single-pixel approach. You will generate a binary mask matrix that the projector will use to illuminate multi-pixel patterns onto your object. Before scanning your custom images, you'll walk through a basic multi-pixel imaging simulation to understand how it works, delve deeper into the differences between ideal and non-ideal imaging, and understand why certain matrices are better than others at imaging in noisy systems.

        *Note: A lot of the code to complete this lab will be provided for you to run. However, looking over the code to try to understand what it does is **highly encouraged**. Additionally, we will be writing **functions** to enable multiple parts of this lab to reuse the same code with minimal copy + pasting.*

        **<span style = "color: red">Run the following code block to get access to several pre-written functions and helper libraries.</span>**
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # <a id = 'matrixGenIntro'><span style = "color: blue">Task 1: Generating Multipixel Scanning Matrices</span></a>

        **Note: This lab will use 0-indexing, as Python uses 0-indexing.**

        Recall that we can define our imaging system by the following mathematical model:

        $$ H \vec{i} = \vec{s} $$ 

        $H$ is the imaging mask matrix, $\vec{i}$ is our image in column vector form, and $\vec{s}$ is the sensor output also in column vector form.

        In Imaging 2, we scanned our image by highlighting one pixel at a time in a mask. Each row $H_k$ of $H$ defined a 1-D representation of each mask. This meant that scanning an image with $n$ pixels would require an $H$ with $n$ rows. Since we project one mask at a time onto our image, we would need to do exactly $n$ scans. Take a $2\times2$ image for example: $H$ would need exactly 4 rows, and we would make 4 scans.

        Let's try something different. We'll still do $n$ scans of our image, but let's try to highlight more than one pixel per mask. We still want to be able to reconstruct our image from our sensor values. So the question is: how do you choose which pixels to illuminate with each mask?

        Begin by assigning each pixel value in the 2x2 image to a variable, $p_{ij}$, where $i$ is the row and $j$ is the column associated with the pixel location. <br/><br/>

        <center>
            <b>2x2 Image</b>
        <img src="public\img_4x4_new.png" align="center" style="height:200px" />
        </center>

        <!--
        In matrix form, the 2x2 image will look like this:
        $$\begin{bmatrix} p_{00} & p_{01} \\ p_{10} & p_{11} \end{bmatrix}$$
        -->

        In our mathematical model above, we represent the 2x2 image as the 1D column vector: 

        $$\vec{i} = \begin{bmatrix} p_{00} \\ p_{01} \\ p_{10} \\ p_{11} \end{bmatrix}$$

        Likewise, the sensor reading column vector is represented as:

        $$\vec{s} = \begin{bmatrix} s_0 \\ s_1 \\ s_2 \\ s_3 \end{bmatrix}$$

        Where the sensor reading from the $k$th mask is $s_k$. In the example above, the sensor reading from the 2nd mask is $s_2$. (We consider $s_0$ to be the 0th mask.)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To illustrate the relationship between the mask matrix $H$ (with per-row imaging masks $H_k$), the image vector $\vec{i}$, and the sensor reading vector $\vec{s}$, we provide you with this **Example System of Linear Equations:**

        $$
        \begin{align} 
        s_0 & = p_{00}\\
        s_1 & = p_{00} + p_{01}\\
        s_2 & = p_{00} + p_{10}\\
        s_3 & = p_{01} + p_{10} + p_{11}
        \end{align}
        $$

        **<span style = "color: red">*IMPORTANT*: The above system of equations is only an example! It only serves an illustrative purpose for this section of the lab. Please do not use it for the rest of the lab.</span>**

        How would you represent the above as a mask matrix $H$? Convince yourself that the image below does just that (where a **white pixel** represents a value of **1** and a **black pixel** represents a value of **0**).</font><br/><br/>

        <center>
            <b>Imaging Mask Matrix $H$ for the Example System of Linear Equations</b>
        <img src="public/mask_sample_4x4.png" align="center" style="height:200px" />
        </center>

        Recall that each row of our mask matrix represents a mask in 1-D form. We must *reshape* each row $H_k$ of $H$ into the 2-D mask (Mask $k$) itself before projecting it onto the image. To make sense of the $H$ matrix, it is helpful to look at each mask individually. Let's consider $H_0$, the 0th row of $H$. When we reshape the 1-D $1\times4$ row into a $2\times2$ mask, we get Mask 0 (below on the left). <br/><br/>

        <center>
            <b>Individual Masks for the Example System of Linear Equations</b>
        <img src="public/H_4x4_split.png" align="center" style="height:200px" />
        </center>

        Now we can see that 

        $$H_k \vec{i} = s_k$$ 

        represents one of the equations in our system. 

        For example, the equation for $s_0$ only depends on one pixel, $p_{00}$, i.e. the top-left pixel of our $2\times2$ image. We can represent it algebraically as:

        $$s_0 = H_0 \vec{i}$$

        where $H_0 = \begin{bmatrix} 1 & 0 & 0 & 0 \end{bmatrix}$.

        So numerically, we can represent the equation as:

        $$s_0 = \begin{bmatrix} 1 & 0 & 0 & 0 \end{bmatrix} \begin{bmatrix} p_{00} \\ p_{01} \\ p_{10} \\ p_{11} \end{bmatrix}$$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## <a id = 'simpleMatrixGen'><span style = "color: blue">Task 1a: Imaging Mask Matrix Practice</span></a>

        Now that we have an understanding of how to approach multipixel scanning, let's test our approach on a new system of equations:

        <center>
        <b>Lab 3 System of Equations</b>

        \[
        \begin{align}
        s_0 & = p_{00} + p_{01} + p_{10}\\
        s_1 & = p_{00} + p_{11}\\
        s_2 & = p_{01} + p_{11}\\
        s_3 & = p_{10} + p_{11}
        \end{align}
        \]

        **<span style="color: red">For a 2x2 image represented by $\vec{i}$, create the matrix H such that $H \cdot \vec{i} = \vec{s}$ represents the $\textbf{Lab 3 System of Equations}$ above.</span>**
        """
    )
    return


@app.cell(hide_code=True)
def _(hint_1, mo):
    mo.md(
        f"""
        {mo.as_html(hint_1)}
        """
    )
    return


app._unparsable_cell(
    r"""
    # TODO: Create H (4x4) for the Lab 3 System of Equations --------------------
    H_new = # YOUR CODE HERE

    # Show H
    plt.imshow(H_new, cmap = \"gray\", interpolation = \"nearest\")
    plt.title(\"4x4 H\")

    # Run autograder
    test_H_new(H_new)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As stated above, we will *reshape* rows $H_k$ of the mask matrix, $H$, into the individual masks themselves. 

        **<span style="color: red">You will help write a function `show_masks` that enables you to iterate through the 4 individual masks and display them as 2x2 images (TODO). Double check that the generated masks make sense visually and have the expected number of illuminated pixels. The `show_masks` function will be reused later.</span>**
        """
    )
    return


@app.cell(hide_code=True)
def _(hint_2, mo):
    mo.md(
        f"""
        {mo.as_html(hint_2)}
        """
    )
    return


app._unparsable_cell(
    r"""
    # Inputs
    #  `H`: Mask matrix
    #  `rows`: Number of rows in image (height)
    #  `cols`: Number of columns in image (width)
    #  `num_masks_shown`: Number of individual masks to display (starting from `H` row 0)
    def show_masks(H, rows, cols, num_masks_shown):
        plt.figure(figsize = (18, 12)) 
        # Use this for loop to iterate through the first `numMasksShown` rows of `H` 
        # you want to display.
        for k in range(num_masks_shown):
            plt.subplot(num_masks_shown, num_masks_shown, k + 1)

            # TODO: Reshape the `k`th row of `H` to be shown in 2D --------------------
            mask = # YOUR CODE HERE

            plt.imshow(mask, cmap = \"gray\", interpolation = \"nearest\")
            # Title also prints number of illuminated (white) pixels per mask
            plt.title(\"Mask \" + str(k) + \": \" + str(np.sum(H[k])) + \" Illuminated Pixels\")
        plt.show()

    # Show individual masks    
    show_masks(H = H_new, rows = 2, cols = 2, num_masks_shown = 4)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## <a id = 'randomBinaryMatrixGen'><span style = "color: blue">Task 1b: Generating a Random Binary Mask Matrix</span></a>

        A 2x2 image is not very interesting to scan, so we will instead try to scan a 32x32 region. Note that this image has different dimensions compared to last week's image!

        **<span style="color: red">To scan a 32x32 image, what dimensions must our scanning matrix $H$ have? What does the number of rows of $H$ correspond to? What does the number of columns correspond to? What do the elements in each column of $H$ represent?</span>**
        """
    )
    return


@app.cell(hide_code=True)
def _(answer_1, mo):
    mo.md(
        f"""
        {mo.as_html(answer_1)}

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since we'd like to use a sufficiently interesting set of masks and you *really* don't want to be constructing such a large matrix by hand, we will provide you with a function that generates a random binary mask matrix $H$ for you, given dimensions `(rows, cols)` corresponding to your image's height/width, and, as we'll go into later, a parameter for the average number of illuminated pixels per scan. The resulting matrix $H$ will consist entirely of 0's and 1's, where 1's are randomly interspersed among 0's, and each row will contain approximately **`avg_1s_per_row`** (see function arguments) # of 1's. Not all rows will contain the same number of 1's!

        **<span style="color: red">Run the `generate_random_binary_mask` function and visually inspect that the generated `randomH` (with approximately 300 pixels illuminated per scan) has the right dimensions & visually looks random. Don't worry too much about how this function is actually implemented, but you can check out the code in `scripts/helpers.py`.</span>**
        """
    )
    return


@app.cell
def _(generate_random_binary_mask):
    # Generate the randomH mask
    randomH = generate_random_binary_mask(avg_1s_per_row = 300)
    return (randomH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color: red">Use the `show_masks` function created earlier to show the first 4 individual masks (rows 0 to 3 of `randomH`) as 32x32 images.</span>**""")
    return


@app.cell
def _():
    # TODO: Reuse the `show_masks` function from earlier to display the first 4 masks of randomH. -------

    # YOUR CODE HERE
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Think back to the representation of the imaging system as taking a matrix-vector product. Recall that in the Imaging 2 lab, you reconstructed the image column vector $\vec{i}$ from the sensor reading vector $\vec{s}$ by applying the equation:

        $$\vec{i} = H^{-1} \vec{s}$$

        You used the **identity** matrix for $H$, for which the inverse $H^{-1}$ exists. In order to apply the same reconstruction method assuming a randomly generated binary $H$, you first need to make sure that your $H$ is actually invertible. 

        **<span style="color: red">What must be true about the rows of $H$ for it to be invertible? What about the columns?</span>**
        """
    )
    return


@app.cell(hide_code=True)
def _(answer_2, mo):
    mo.md(f"""{mo.as_html(answer_2)}""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Note on invertibility**

        Luckily, randomly generated binary matrices are *usually* invertible. However, the function we provided still double checks that the generated $H$ is indeed invertible (using an alternative method to Gaussian elimination), and re-generates the matrix if it's not.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # <a id = 'simulatorIntro'><span style = "color: blue">Task 2: Imaging Simulator</span></a>

        Let's pause for a minute before we start capturing images with the projector. Recall from Imaging 2 that the projector setup is usually placed inside a cardboard box to prevent light from the outside world disturbing our sensor. Even when the projector is turned off, there might be a significant amount of light inside the box. The sensor and related circuits generate noise in our measurements due to thermal physics and other environmental factors. The refresh rate of the projector also contributes to noise. Thus, non-idealities like noise will inevitably be present in our setup. This is a limitation of the real-world setup that greatly affects our ability to reconstruct the image using the light sensor data. That's why it's important to build a simulator that **accurately models** what happens when we try to capture an image, including non-idealities that we can potentially compensate for.

        Our virtual simulated projector will artificially generate noise to affect sensor results in a way that mirrors this real-world phenomenon.

        ## <a id = 'idealSensor'><span style = "color: blue">Task 2a: Constructing an Ideal Sensor Model</span></a>

        Let's first construct a function that emulates what we would *hope* occurs when we scan an image (ideal imaging). An image (represented as the column vector $\vec{i}$) is placed in a region that can be illuminated by the projector. The projector projects a sequence of masks $H_k$ onto the image (illuminating certain pixels at a time). In our simulation, the digitized 'light sensor' output is the sum of the brightnesses detected across illuminated pixels. The $k^{th}$ entry of the sensor output vector, $s_k$, corresponds to the $k^{th}$ scan.

        Recall that these operations can be represented by the previously defined mathematical model:

        $$\vec{s} = H \vec{i}$$ 

        **<span style="color:red"> Your first goal is to translate this ideal model into a `simulate_ideal_capture` function (Fill in the TODO). Apply the function using the supplied 32x32 image of a playing card and your generated random binary matrix `H`. Display the simulated sensor reading as a 32x32 image.</span>**

        The playing card you're trying to image should look like: <br/><br/>

        <center>
        <img src="public/raw_card.png" align="center" style="height:200px" />
        </center>

        Think about what the output sensor readings will look like. Given randomly generated masks, would you expect the output sensor readings to be remotely recognizable?
        """
    )
    return


@app.cell(hide_code=True)
def _(hint_3, mo):
    mo.md(
        f"""
        {mo.as_html(hint_3)}
        """
    )
    return


app._unparsable_cell(
    r"""
    # Inputs: 
    #  `i2D`: 2D image you're trying to capture
    #  `H`: Mask matrix
    #  `matrixName`: Name of mask matrix (for image title)
    #  `display`: Whether to display the sensor output as a 2D image
    # Outputs:
    #  `s`: Sensor reading column vector
    def simulate_ideal_capture(i2D, H, matrix_name, display = True):
        # Number of pixels in your image = `iHeight` * `iWidth`
        i_height = i2D.shape[0]
        i_width = i2D.shape[1]
        i_size = i_height * i_width

        # TODO: Convert the 2D image `i2D` into a 1D column vector `i`
        i = # YOUR CODE HERE

        # TODO: Perform the matrix operation to emulate the ideal imaging system  --------------
        s = # YOUR CODE HERE

        if display:
            # Reshape the simulated sensor output `s` into an appropriately 
            # sized 2D matrix `s2D` and plots it
            s2D = np.reshape(s, (i_height, i_width))
            plt.imshow(s2D, cmap = \"gray\", interpolation = \"nearest\")
            plt.title(\"Ideal Sensor Output, Using %s\" % matrix_name)
            plt.show()
        return s

    # Load playing card image + display it
    i2D = np.load(\"scripts/raw_card.npy\")
    plt.imshow(i2D, cmap = \"gray\", interpolation = \"nearest\")
    plt.title(\"Raw 32x32 Image of the playing card\")
    plt.show()

    # Simulate the image capture step (ideal)
    s = simulate_ideal_capture(i2D = i2D, H = randomH, matrix_name = \"Random H\");
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### <a id = 'idealReconstruction'><span style = "color: blue">Image Reconstruction Using the Ideal Sensor Model + Matrix Inverse</span></a>

        As you can see, for *multipixel imaging*, the sensor output does not resemble the original image in any way. By applying the randomly generated mask matrix $H$, you've essentially encrypted the image data, making it unrecognizable to anyone who doesn't know the exact mask matrix $H$ you used (otherwise known as the encryption key).

        If you know the key $H$, as stated before, you can reconstruct/decrypt the desired image column vector $\vec{i}$ from the sensor reading vector $\vec{s}$ by essentially *undoing* what the imaging system did to the image and applying the equation:

        $$\vec{i} = H^{-1} \vec{s}$$

        Again, it's important that we've selected an invertible $H$. 

        **<span style="color:red">Now your job is to help write a function `ideal_reconstruction` (Fill in the TODO) that accepts the column vector $\vec{s}$ and mask matrix $H$ and displays the reconstructed estimate of $\vec{i}$ as a 2D image. Run the reconstruction function using the previously computed `s` and mask matrix `H` and verify that it worked as you expected.</span>**
        """
    )
    return


@app.cell(hide_code=True)
def _(hint_4, mo):
    mo.md(
        f"""
        {mo.as_html(hint_4)}
        """
    )
    return


app._unparsable_cell(
    r"""
    # Inputs
    #  `H`: Mask matrix
    #  `matrix_name`: Name of mask matrix (for image title)
    #  `s`: Sensor reading column vector
    #  `rows`: Number of rows in image (height)
    #  `cols`: Number of columns in image (width)
    def ideal_reconstruction(H, matrix_name, s, rows = 32, cols = 32, real_imaging = False):

        # TODO: Perform the matrix operations required for reconstruction --------------------
        i = # YOUR CODE HERE

        if real_imaging:
            i = noise_massage(i, H)

        # Reshape the column vector `i` to display it as a 2D image
        i2D = # YOUR CODE HERE  

        # We're going to exclude the top row and left-most column from display
        plt.imshow(i2D[1:, 1:], cmap = \"gray\", interpolation = \"nearest\")
        plt.title(\"Reconstructed Image, Using %s\" % matrix_name)
        plt.show()


    # Run ideal reconstruction    
    ideal_reconstruction(H = randomH, matrix_name = \"Random H\", s = s)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## <a id = 'nonidealities'><span style = "color: blue">Task 2b: Handling System Non-Idealities</span></a>

        The ideal reconstruction demonstrated above works great, right? Unfortunately, due to real-world non-idealities alluded to earlier, if you directly tried to image a drawing with the multipixel masks in $H$, the reconstruction would probably look terrible. A significant amount of engineering effort is focused on how to best translate theory into practice by attempting to compensate for or remove non-idealities. In the following sections, we'll look at some of the worst offenders and what we can do to improve reconstruction quality. 

        ### <a id = 'noiseSimulation'><span style = "color: blue">Noise *(Why So Grainy? ☹)*</span></a>

        We will see noise again later in Module 3,-- if you're really interested in modelling noise and its effects, it's covered more extensively upper division classes including EE123, EE126, EE142 -- but for now it's important to realize that both the light sensor circuit and the projector add noise that shows up in the digitized sensor output. Noise is what causes photos to look grainy or fuzzy. As an example, if your single pixel imaging system from last week happened to be very noisy (and usually, the cheaper the system, the noisier it is...), imaging the playing card from before might've produced something like: <br/><br/>

        <center>
        <img src="public/noisy_card.png" align="center" style="height:200px" />
        </center>

        The noisier your system, the less the resultant image will look like what you expected ☹.

        One way to make noise less problematic is to increase the number of pixels illuminated per scan. This increases the "signal level" (i.e. contributions from things we actually care about). At the same time, the amount of noise coming from the light sensor circuit and projector should stay mostly constant, thus improving the so-called *signal-to-noise ratio* (SNR) of our system. This is important to know when choosing `avg_1s_per_row` for our random binary mask.

        However, in reality, the number of pixels illuminated per scan is limited by the ambient light sensor circuit. This is because at high brightness levels, the sensor circuit becomes saturated. The difference in sensor reading for each additional illuminated pixel becomes quite small, and thus we lost the ability to differentiate the number of illuminated pixels.

        Another way to make noise less problematic is to repeat each scan $k$ (with the same illumination pattern) many times and *average* the sensor outputs. The desired signal is always present, but the *random* error (noise) changes on each repeat scan. Thus, you can "average out the noise" at the expense of spending more time acquiring the image. This is actually what the Arduino code used in lab does under the hood.

        Generally speaking, we would like to build a sensing system that is as noise robust as possible, but what does that entail?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## <a id = 'eigenanalysis'><span style = "color: blue">Task 2c: Eigenanalysis &amp; the Robustness of Inverse-Based Reconstruction</span></a>

        ## <span style = "color: red">THIS SECTION IS VERY IMPORTANT. PLEASE READ CAREFULLY.</span>
        When noise is included, the mathematical model of our imaging system would look like:

        $$ \vec{s} = H \vec{i} + \vec{\omega} +\vec{o} $$

        The vector $\vec{o}$ is a vector of equal entries, which represents a constant offset from extra light from the projector while it is projecting the color black. Even though black is supposed to be an absence of light, there is still a glow present from the projector that can offset our measurement by a scalar amount. This needs to be removed, but can easily be done so by measuring and subtracting it. 

        The elements ($\omega_k$) of the column vector $\vec{\omega}$ correspond to the random amounts of noise added at each measurement $s_k$. We cannot remove noise, but we can try to reduce its effects.

        For example, you might expect your sensor readings $\vec{s}$ to be something like 

        \begin{equation}
        \vec{s_{expected}} = \begin{bmatrix}
        51 \\
        65 \\
        42 \\
        \vdots \\
        32
        \end{bmatrix}
        \end{equation}

        But you may get something like

        \begin{equation}
        \vec{s_{reality}} = \begin{bmatrix}
        61.2 \\
        76.0 \\
        51.7 \\
        \vdots \\
        44.0
        \end{bmatrix}\;.
        \end{equation}

        This means that what you are getting is really

        \begin{equation}
            \vec{s_{reality}} = \vec{s_{expected}} \;+\; \vec{\omega} \;+\; \vec{o}
        \end{equation}

        \begin{equation}
            \begin{bmatrix}
                61.2 \\
                76.0 \\
                51.7 \\
                \vdots \\
                44.0
            \end{bmatrix} = 
            \begin{bmatrix}
                51 \\
                65 \\
                42 \\
                \vdots \\
                32
            \end{bmatrix}
            \;+\;
            \begin{bmatrix}
                0.2 \\
                1.0 \\
                -0.3 \\
                \vdots \\
                2.0
                \end{bmatrix}
            \;+\;
            \begin{bmatrix}
                10 \\
                10 \\
                10 \\
                \vdots \\
                10
            \end{bmatrix}
        \end{equation}

        where the last two vectors are $\vec{\omega}$ and $\vec{o}$

        As you can see, once you measure the offset, it is very easy to just subtract from your measurements. We will take care of this for you in the experimental portion of the lab (below), so you don't need to worry about it. We will ignore it in the rest of the notebook, and assume it is subtracted.

        From this point forward, our key equation will look like this:

        $$ \vec{s} = H \vec{i} + \vec{\omega}$$

        Now we will try to reconstruct the image $\vec{i}$ with matrix inversion $H^{-1}$:

        $$ H^{-1}\vec{s} = H^{-1}H \vec{i} + H^{-1}\vec{\omega}$$

        $$ H^{-1}\vec{s} = \vec{i} + H^{-1}\vec{\omega}$$

        We will call $H^{-1}\vec{s} = \vec{i_{est}}$ leaving us with 
        $$\vec{i_{est}} = H^{-1} \vec{s} = \vec{i} + H^{-1} \vec{\omega}$$

        Remember that we were hoping to solve for just $\vec{i}$. The additional undesired term $H^{-1} \vec{\omega}$ is what we call our reconstruction *error*, which results from linearly transforming the original noise vector $\vec{\omega}$ by $H^{-1}$. This implies that our choice of $H$ (and therefore $H^{-1}$) strongly influences how robust our overall imaging system is.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To build some intuition on why this is the case, recall that matrix-vector multiplication $A \vec{x} = \vec{b}$ linearly transforms $\vec{x}$ into $\vec{b}$ via scaling and rotation, as designated by $A$. Additionally, recall that the eigenvalues $\lambda_i$ and $N$ length eigenvectors $\vec{v_{\lambda_i}}$ of an $N \times N$ matrix $A$ can be found by solving for:

        $$A \vec{v_{\lambda_i}} = \lambda_i \vec{v_{\lambda_i}}$$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Multiplying both sides of this equation by $A^{-1}$ and dividing by $\lambda_i$ allows us to rewrite this equation as:

        $$A^{-1} \vec{v_{\lambda_i}} = \frac{1}{\lambda_i} \vec{v_{\lambda_i}} $$ 

        How does this help reconstruct our image? We know that $H$, which is an $N \times N$ matrix, is invertible, and thus we know it has at most $N$ linearly independent eigenvectors. 

        Our matrix $H$ also has another property, which we haven't learned about yet: it is **diagonalizable**. Diagonalizable matrices are beyond the scope of this course (but covered in EECS16B). For now, all you need to know is that a diagonalizable $N \times N$ matrix $H$ has *precisely* N linearly-independent eigenvectors.

        So if we know $H$ has N eigenvectors, we know they span $\mathbb{R}^N$. In other words, the eigenvectors of $H$ form a basis for $\mathbb{R}^N$. Well, guess what? Our noise vector lies in $\mathbb{R}^N$. So we can write it as a linear combination of the eigenvectors like so:

        $$\vec{\omega} = \alpha_1 \vec{v_1} + ... + \alpha_N \vec{v_N}$$

        Now if we apply $H^{-1}$ to both sides of the equation,

        $$H^{-1} \vec{\omega} = H^{-1} \alpha_1 \vec{v_1} + ... + H^{-1}\alpha_N \vec{v_N}$$

        Pull out the $\alpha$ constants in front of $H^{-1}$ since scalars commute with matrices

        $$H^{-1} \vec{\omega} = \alpha_1 H^{-1} \vec{v_1} + ... + \alpha_N H^{-1} \vec{v_N}$$

        And we can apply the eigenvector identity shown above:

        $$H^{-1} \vec{\omega} = \alpha_1 \frac{1}{\lambda_1} \vec{v_1} + ... + \alpha_N \frac{1}{\lambda_N} \vec{v_N}$$

        So we can see that regardless of the scaling constants $\alpha$, if we have very large eigenvalues of $H$ then each component of $\vec{\omega}$ is attenuated, and likewise if each eigenvalue is small, the noise vector will be amplified.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### <a id = 'graphicalInterpretation'><span style = "color: blue">Graphical Interpretation</span></a>

        Another way we can picture this is by showing a graphical example, thinking of how $H$ is a transformation that rotates and scales vectors. In the following image, we have our ideal sensor readings, $H\vec{i}$ and a noise vector, $\vec{\omega}$. After applying two different matrices, $H_1^{-1}$ and $H_2^{-1},$ we can see how each vector is transformed. Ideally we would want the $\vec{\omega}$ vector to be $\vec{0}$, so the recovered image is the same as the ideal reconstruction. Adding everything together to get the final result, we have $\vec{i}+H^{-1}\vec{\omega}$. Depending on the choice of $H$, the noise may end up amplified or attenuated.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <center><img src="public/2d_transform.jpg" align="center"/>
            <b>Visual representation of the effect of different matrices on the noise vector $\omega$</b>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Numerically, we can also see what happens to the noise with different matrices by applying them to a given noise vector. We will now introduce a special matrix called the <a href="https://mathworld.wolfram.com/HadamardMatrix.html" target="_blank">Hadamard matrix</a>. It has interesting properties useful in many applications. **The code below prints out the magnitude (norm) of the noise vectors after applying the inverses of the random masking matrix and the Hadamard matrix.**""")
    return


@app.cell
def _(create_hadamard_matrix, generate_random_binary_mask, np):
    randomH_1 = generate_random_binary_mask(avg_1s_per_row=300, plot=False)
    hadamardH = create_hadamard_matrix(shape=randomH_1.shape, plot=False)
    sigma = 7
    noise = np.random.normal(0, sigma, randomH_1.shape[0])
    hadamard_norm = np.linalg.norm(np.dot(np.linalg.inv(hadamardH), noise))
    random_norm = np.linalg.norm(np.dot(np.linalg.inv(randomH_1), noise))
    print('Norm of the noise vector after hadamardH inverse: ', hadamard_norm)
    print('Norm of the noise vector after randomH inverse: ', random_norm)
    return hadamardH, hadamard_norm, noise, randomH_1, random_norm, sigma


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">Which matrix amplifies the noise less?</span>**""")
    return


@app.cell(hide_code=True)
def _(answer_3, mo):
    mo.md(f"""{mo.as_html(answer_3)}""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### <a id = 'revisitingIdentity'><span style = "color: blue">Revisiting the Identity Matrix</span></a>
        We know that the identity matrix is invertible, but is it a good masking matrix? To answer that question, we need to know its eigenvalues.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">What are the eigenvalues of the Identity matrix? What are its eigenvalues if we scale the identity matrix by a constant? What are its eigenvectors?</span>**""")
    return


@app.cell(hide_code=True)
def _(answer_4, mo):
    mo.md(f"""{mo.as_html(answer_4)}""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Scaling the eigenvalues of the identity matrix essentially boils down to "dimming" the simulated sensor readings, or making them "brighter." Think about how good your scan would be if the projector only operated on 1%, or conversely 100% of its max light intensity. It is unlikely that both would give you the same quality sensor readings. 

        **Run the next block to show the ideal image, and the noise that gets added to the image. Change the constant that scales the identity from low values like 0.1 to large values like 100 to see how the noise changes with increasing or decreasing eigenvalues.**

        Note that the noise is visualized in a different color scale than black and white.
        """
    )
    return


@app.cell
def _(np, plot_image_noise_visualization):
    scale_factor = 1
    i2D = np.load('scripts/raw_card.npy')
    (M, N) = i2D.shape
    H = scale_factor * np.eye(M * N)
    sigma_1 = 1.25
    noise_1 = np.random.normal(0, sigma_1, H.shape[0])
    noise_1 = np.reshape(noise_1, (M, N))
    s = H.dot(i2D.ravel()).reshape((M, N)) + noise_1
    recovered_image = np.linalg.inv(H).dot(s.ravel()).reshape((M, N))
    plot_image_noise_visualization(i2D, noise_1, s, H)
    return H, M, N, i2D, noise_1, recovered_image, s, scale_factor, sigma_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">Which scaling factor performs better: 0.01 or 1000? Why?</span>**""")
    return


@app.cell(hide_code=True)
def _(answer_5, mo):
    mo.md(f"""{mo.as_html(answer_5)}""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### <a id = 'comparingScanning'><span style = "color: blue">Comparing Scanning Matrices</span></a>
        Now let's take a look at the two matrices we will use to scan, hadamardH and randomH. The block of code below will show the ideal recovered image, along with the noise that gets added on top, and the total result. We will take care of generating the noise--all you have to do is tell us how much noise to add (by setting the `noise_magnitude` variable). In addition to displaying the images, the code will also print out the norm of the modified noise vector $H^{-1}\vec{\omega}$ so you can see quantitatively how different matrices impact the noise.

        [comment]: <> (**<span style="color:red">First, just run the next code block so that you'll have access to `simulateCaptureWithNoise` below.</span>**)
        **<span style="color:red">You will simulate the imaging system with different amounts of noise added. Run the code block below and change the noise magnitude to see how the output is affected.</span>**
        """
    )
    return


@app.cell
def _(
    create_hadamard_matrix,
    generate_random_binary_mask,
    np,
    plot_image_noise_visualization,
    s,
):
    noise_magnitude = 50.0
    i2D_1 = np.load('scripts/raw_card.npy')
    (M_1, N_1) = i2D_1.shape
    randomH_2 = generate_random_binary_mask(avg_1s_per_row=300, plot=False)
    hadamardH_1 = create_hadamard_matrix(shape=randomH_2.shape, plot=False)
    sigma_2 = noise_magnitude / np.sqrt(M_1 * N_1)
    noise_2 = np.random.normal(0, sigma_2, randomH_2.shape[0])
    noise_2 = np.reshape(noise_2, (M_1, N_1))
    plot_image_noise_visualization(i2D_1, noise_2, s, randomH_2, title='Reconstruction with Random $H$')
    modified_noise_norm = np.linalg.norm(np.linalg.inv(randomH_2).dot(noise_2.ravel()))
    print('Norm of Hinv*w = %0.4f' % modified_noise_norm)
    plot_image_noise_visualization(i2D_1, noise_2, s, hadamardH_1, title='Reconstruction with Hadamard $H$')
    modified_noise_norm = np.linalg.norm(np.linalg.inv(hadamardH_1).dot(noise_2.ravel()))
    print('Norm of Hinv*w = %0.4f' % modified_noise_norm)
    return (
        M_1,
        N_1,
        hadamardH_1,
        i2D_1,
        modified_noise_norm,
        noise_2,
        noise_magnitude,
        randomH_2,
        sigma_2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">What noise magnitudes did you have to use for each of the two matrices to make the image borderline unrecognizable?</span>**""")
    return


@app.cell(hide_code=True)
def _(answer_6, mo):
    mo.md(f"""{mo.as_html(answer_6)}""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For this next section, we will examine the Hadamard matrix in a bit more detail. We will use the function `eigen_analysis_comparison` that plots a histogram of the magnitudes of the eigenvalues of your $H$'s and their respective inverses (x axis = magnitude bins, y axis = number of eigenvalues within the bin's magnitude range).""")
    return


@app.cell
def _(eigen_analysis_comparison, hadamardH_1, np, randomH_2):
    eigen_analysis_comparison(H1=randomH_2, matrix_name_1='Random Binary H', H2=hadamardH_1, matrix_name_2='Hadamard H')
    randomH_inv = np.linalg.inv(randomH_2)
    hadamardH_inv = np.linalg.inv(hadamardH_1)
    eigen_analysis_comparison(H1=randomH_inv, matrix_name_1='Inverse of Random Binary H', H2=hadamardH_inv, matrix_name_2='Inverse of Hadamard H')
    return hadamardH_inv, randomH_inv


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">Which of the two matrices `randomH` and `hadamardH` do you think is more noise robust and would result in a better reconstruction? Justify your answer using the eigenvalue histograms above.</span>**""")
    return


@app.cell(hide_code=True)
def _(answer_7, mo):
    mo.md(f"""{mo.as_html(answer_7)}""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='scanningImages'></a>
        # <span style="color:blue">Task 3: Scanning Images</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## <a id ='realImaging'><span style = "color: blue">Real Multipixel Imaging</span></a>

        In the previous section, we scanned our image one pixel at a time. Now we are going to use the two matrices you examined earlier to scan.

        Let's start with the random binary mask.
        """
    )
    return


@app.cell
def _(generate_random_binary_mask):
    randomH_3 = generate_random_binary_mask(avg_1s_per_row=300, plot=False)
    sr_1 = []
    return randomH_3, sr_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">Let's reconstruct your image. Based off of your simulation results, is this the reconstruction quality that you expected using `H`? Think about how noisy our actual imaging system is.</span>**""")
    return


@app.cell(hide_code=True)
def _(hint_6, mo):
    mo.md(
        f"""
        {mo.as_html(hint_6)}
        """
    )
    return


@app.cell
def _(randomH_3, reconstruct_multipixel, sr_1):
    reconstruct_multipixel(randomH_3, sr_1, width=32, height=32)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **As expected, the randomly generated H matrix does not work well, if at all.** Please don't worry if your scan is not ideal!

        Next, let's try to image your index card using `hadamardH`. Imaging with `hadamardH` requires some additional pre-processing to stitch sensor readings (associated with the mask that we split) back together. This has been taken care of for you in the code below. 

        **<span style="color:red">Run the following code block. It will capture sensor readings using the Hadamard matrix `hadamardH`.</span>**
        """
    )
    return


@app.cell
def _(create_hadamard_matrix):
    hadamardH_2 = create_hadamard_matrix(shape=(32 * 32, 32 * 32), plot=False)
    sr_2 = []
    return hadamardH_2, sr_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<span >Don't worry if your sensor output "looks" incorrect. hadamardH is different from other matrices we've used before. We must reconstruct the image to check correctness. </span>""")
    return


@app.cell
def _(hadamardH_2, reconstruct_multipixel, sr_2):
    reconstruct_multipixel(hadamardH_2, sr_2, width=32, height=32)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">Let's reconstruct your image. Based off of your simulation results, is this the reconstruction quality that you expected using `hadamardH`?</span>**""")
    return


@app.cell(hide_code=True)
def _(answer_7, mo):
    mo.md(f"""{mo.as_html(answer_7)}""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">Comment on your reconstruction results when using `randomH` and `hadamardH`. In real imaging, which matrix did better? Did this match your expectations from simulation? Why? How did you expect multipixel imaging to compare to single pixel imaging from Imaging 2? What are some observed limitations of multipixel imaging?</span>**""")
    return


@app.cell(hide_code=True)
def _(answer_7, mo):
    mo.md(f"""{mo.as_html(answer_7)}""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='useCases'></a>
        # <span style="color:blue">Task 4: Understanding Multipixel Use-Cases</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Based on the results you got in Task 3, you may be wondering why we'd ever use multipixel imaging at all. After all, we've significantly increased the complexity of our masks and reconstruction procedure only to produce images that don't seem to be any better (or are potentially even worse!) than before.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To highlight the advantages of multipixel imaging, let's return to our discussion of our mathematical model:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""$$ \vec{s} = H \vec{i}_{est}$$""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""$\vec{s}$ can be interpreted as a vector of measurements: every time we project a mask onto our image and record the sensor data, it's stored as an element of $\vec{s}$. As we've seen before, solving for $\vec{i}$ yields:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""$$H^{-1}\vec{s} = \vec{i}_{est}$$""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If $H$ corresponds to a single-pixel setup, then 'unscrambling' (or 'inverting') $H$ will simply involve rearranging the order of elements in $\vec{s}$ (or leaving it alone if $H$ is the identity matrix). In other words, if each row of $H$ has exactly one 1 with the remaining entries equal to 0, then the same will be true of $H^{-1}$. **Because of this, with single-pixel imaging, each value in $\vec{i}_{est}$ depends on exactly one measurement!**""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""However, this reasoning doesn't apply to multipixel imaging. If $H$ has multiple nonzero values in each row, then we expect $H^{-1}$ to behave similarly. **As such, with multipixel imaging, each value in $\vec{i}_{est}$ can depend on several measurements.** This makes multipixel imaging more robust in certain circumstances.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**For example, what happens if we're in a scenario where the power temporarily goes out during our scan?** In this case, we've essentially ruined a small set of our measurements. This is where we'll see a big difference between single and multipixel imaging: in single-pixel imaging, the pixels corresponding to those measurements will be irrecoverably lost, but we'll still get a decent image with multipixel imaging since each pixel's value is distributed over multiple measurements.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To illustrate this, we used the setup discussed in this lab to image this drawing:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<center><img src="public/camerashot.jpg" align="center" style="height:200px"/>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To ruin some of our measurements, we simply opened the carboard box for ten seconds during our scans. Here's the result from the single-pixel scan:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<center><img src="public/singleboxopen10s.png" align="center" style="height:200px"/>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Although you can make out most of the image, the pixels during which the box was open (the white strip) were clearly lost. Overall, this isn't a great picture. Let's see what happens when we do this with the Hadamard matrix instead:""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""<center><img src="public/hadamardboxopen10s.png" align="center" style="height:200px"/>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We opened the box at the same time relative to the single-pixel scan, but we've retained at least some information on most of our pixels, producing a much better image. This is because the noise is distributed over the whole image rather than just over a certain number of pixels. **As is often the case in engineering, we've encountered a tradeoff: would we rather have some pixels recovered perfectly and others not so much, or should we compromise on overall image quality so that we at least get *some* information about *every* pixel?** This is the tradeoff between single and multipixel imaging!""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**<span style="color:red">In your own words, when would you choose to use multipixel imaging instead of single-pixel imaging?</span>**""")
    return


@app.cell(hide_code=True)
def _(answer_8, mo):
    mo.md(f"""{mo.as_html(answer_8)}""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='feedback'></a>
        ## Feedback
        If you have any feedback to give the teaching staff about the course (lab content, staff, etc), you can submit it through this Google form. Responses are **fully anonymous** and responses are actively monitored to improve the labs and course. Completing this form is **not required**.

        [Anyonymous feedback Google form](https://docs.google.com/forms/d/e/1FAIpQLScWVsuuiUC1NAqkhDiTtwmpt68Dy9N29rNiioluqo1CdngFNQ/viewform?usp=header)

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
        When you are ready to get checked off, fill out the checkoff google form. **[Checkoff Form](https://docs.google.com/forms/d/e/1FAIpQLSdIwjFcVYsHI8tfn3NtJD9MeJYvT0VmweR0smPgxHjYbNF22w/viewform?usp=header)**

        Your GSI or a Lab Assistant will join you when they are available and go through some checkoff questions with your group. They will go through the checkoff list in order. Please be patient!
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    from scripts.helpers import generate_random_binary_mask, create_hadamard_matrix, plot_image_noise_visualization, eigen_analysis_comparison, reconstruct_multipixel
    return (
        create_hadamard_matrix,
        eigen_analysis_comparison,
        generate_random_binary_mask,
        mo,
        plot_image_noise_visualization,
        reconstruct_multipixel,
    )


@app.cell
def _(mo):
    # HINTS
    hint_1 = mo.accordion({"Hint": "Look up how to use the `np.array` function."})
    hint_2 = mo.accordion({"Hint": "Reference your code from the part of Imaging 2 where you checked to make sure that the scanning matrix was producing the correct pattern by displaying each of the individual masks. You might want to check out the command `np.reshape`."})
    hint_3 = mo.accordion({"Hint": "Remember to use `np.dot` to do matrix multiplication."})
    hint_4 = mo.accordion({"Hint": "Use np.linalg.inv to invert a matrix."})
    hint_5 = mo.accordion({"Hint": "Because `HSingle` is a special matrix, technically you do not need to perform any matrix operations."})
    hint_6 = mo.accordion({"Hint": "This is NOT supposed to work well! Can you think of why?"})
    return hint_1, hint_2, hint_3, hint_4, hint_5, hint_6


@app.cell
def _(mo):
    answer_1 = mo.ui.text_area(placeholder="Type your answer here ...")
    answer_2 = mo.ui.text_area(placeholder="Type your answer here ...")
    answer_3 = mo.ui.text_area(placeholder="Type your answer here ...")
    answer_4 = mo.ui.text_area(placeholder="Type your answer here ...")
    answer_5 = mo.ui.text_area(placeholder="Type your answer here ...")
    answer_6 = mo.ui.text_area(placeholder="Type your answer here ...")
    answer_7 = mo.ui.text_area(placeholder="Type your answer here ...")
    answer_8 = mo.ui.text_area(placeholder="Type your answer here ...")
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
