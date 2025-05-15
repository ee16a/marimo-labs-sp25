import marimo

__generated_with = "0.11.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Voice Recognition Lab 1: SVD/PCA Classification

        ### EECS 16A: Designing Information Devices and Systems I, Fall 2024


        Junha Kim, Jessica Fan, Savit Bhat, Jack Kang (Fall 2024).


        This lab was heavily inspired by previous EECS16B lab 8, written by Nathaniel Mailoa, Emily Naviasky, et al.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Lab 1: SVD/PCA

        * [Task 1: Data Preprocessing](#part2)
        * [Task 2: PCA via SVD](#task3)
        * [Task 3: Clustering Data Points](#task4)
        * [Task 4: Testing your Classifier](#task5)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='intro'></a>
        ## <span style="color:navy">Introduction</span>

        Throughout lectures and discussion, you have learned theoretical concepts of the DFT and SVD. In the voice recognition lab module, we'll be applying these concepts from class to build an audio classifier using Mel-Scaled Short-Time Fourier transform (STFT) and Principal Component Analysis (PCA), an application of the Singular Value Decomposition (SVD).

        This lab takes in voice recordings, sampling the waveform of an analog signal into discrete points. Multiple syllables will generate multiple peaks and troughs, and soft syllables will have discrete points of lower magnitude than hard syllables (eg. words like "here" and "pear" will have very similar voice recordings)

        In this module, you will build a voice classifier for words of different syllables and intonation in two parts:
        - In lab 1, we will explore dimensionality reduction and classification of voice signals using PCA.
        - In lab 2, we will extend our previous scheme by preprocessing our audio signal with the STFT (very similar to spectrograms that you have seen in the Shazam lab) and feed that as an input to our previous PCA classifier.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Overview of Classification Procedure
        Below is an overview of the classification procedure we will be following in this lab.
        1. Collect recordings of different words. This will form our data set.
        2. Split the data into 2 sets: training and testing data
        3. Preprocess the data to align the words.
        4. Perform SVD and evaluate on the training data split.
        5. Classify each word using clustering in the PCA basis.
        6. Evaluate performance of your PCA model by running the model on testing data.
        7. Make sure you (and your GSI) are satisfied with the classifier's accuracy.

        ### Side Note: Datasets in Machine Learning Applications
        It is common practice in machine learning applications to split a dataset into a training set and testing set (some common ratios for train:test are 80:20 or 70:30). In this lab, we will split 70:30.
        """
    )
    return


@app.cell
def _():
    #import sounddevice
    #import pyaudio
    from tqdm.notebook import trange
    from IPython import display
    #import wave
    import numpy as np
    import csv
    import time
    import matplotlib.pyplot as plt
    import scipy.io
    import utils
    from mpl_toolkits.mplot3d import Axes3D
    cm = ['blue', 'red', 'green', 'orange', 'black', 'purple']
    # '%matplotlib inline' command supported automatically in marimo
    return Axes3D, cm, csv, display, np, plt, scipy, time, trange, utils


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='part2'></a>
        ## <span style="color:navy">Task 1: Data Preprocessing</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Different recordings of the same word can look wildly different, depending on factors like when you started saying the word and how quickly you said it (assuming you are not a robot that can repeat the word 40 times exactly the same way). Thus, before we can use the recorded data for PCA, we must first process the data. We will do this according to the following procedure:
        1. Split the dataset into a training dataset and a test dataset
        2. Align the audio recordings
        3. Apply an envelope (low pass filter) to the recordings to remove irrelevant high-frequency components
        """
    )
    return


@app.cell
def _():
    # YOUR CODE HERE: fill in the blank for the word you just recorded
    all_words_arr = ['jack', 'jason', 'jessica', 'entanglement']
    return (all_words_arr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Task 1a: Align Audio Recordings
        Let's begin by splitting our data into a training and testing set with a 70/30 split. Run the code below to do so.
        """
    )
    return


@app.cell
def _(all_words_arr, np, utils):
    train_test_split_ratio = 0.7
    train_dict = {}
    test_dict = {}
    for _i in range(len(all_words_arr)):
        word_raw = utils.read_csv('{}.csv'.format(all_words_arr[_i]))
        (_word_raw_train, _word_raw_test) = utils.train_test_split(word_raw, train_test_split_ratio)
        train_dict[all_words_arr[_i]] = _word_raw_train
        test_dict[all_words_arr[_i]] = _word_raw_test
    num_samples_train = min(list(map(lambda x: np.shape(x)[0], train_dict.values())))
    num_samples_test = min(list(map(lambda x: np.shape(x)[0], test_dict.values())))
    for (key, raw_word) in train_dict.items():
        train_dict[key] = raw_word[:num_samples_train, :]
    for (key, raw_word) in test_dict.items():
        test_dict[key] = raw_word[:num_samples_test, :]
    return (
        key,
        num_samples_test,
        num_samples_train,
        raw_word,
        test_dict,
        train_dict,
        train_test_split_ratio,
        word_raw,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's take a look at all the training samples.

        **<span style="color:red">Important: It's okay if the recordings aren't aligned. The code in the next part will align the data.</span>**
        """
    )
    return


@app.cell
def _(all_words_arr, plt, train_dict):
    _word_number = 0
    selected_words_arr = all_words_arr
    for _word_raw_train in train_dict.values():
        plt.plot(_word_raw_train.T)
        plt.title('Training sample for "{}"'.format(selected_words_arr[_word_number]))
        _word_number = _word_number + 1
        plt.show()
    return (selected_words_arr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Task 1b: Align Audio Recordings

        As seen above, the speech is a fraction of the 3 second window, and each sample starts at different times. PCA is not good at interpreting delay, so we need to align the recordings and trim to a smaller segment of the sample where the speech is present. To do this, we will use a thresholding algorithm.

        First, we define a **`threshold`** relative to the maximum value of the data. We say that any signal that crosses the threshold is the start of a speech command. In order to not lose the first couple samples of the speech command, we say that the command starts **`pre_length`** samples _before_ the threshold is crossed. We then use a window of the data that is **`length`** long, and try to capture the entire command in that window.

        **Play around with the parameters `length`, `pre_length` and `threshold`** in the cells below to find appropriate values corresponding to your voice and chosen commands. You should see the results and how much of your command you captured in the plots generated below.

        We also pass in a `envelope=True` argument to our `process_data` function: this filters out the frequencies higher than our vocal range, to get back an "envelope" of the waveform of our voice recordings. Why might this be useful?

        Also note that we're "normalizing" the signal towards the end--this is very common in a lot of signal processing applications; another name for this is "min-max feature scaling," where rescale the amplitude range of the signal to be in $[0,1]$ by dividing the signal by the maximum absolute value. This is different from taking a norm of a vector as we've seen in class.
        """
    )
    return


app._unparsable_cell(
    r"""
    def align_recording(recording, length, pre_length, threshold, envelope=False):
        \"\"\"
        align a single audio samples according to the given parameters.

        Args:
            recording (np.ndarray): a single audio sample.

            length (int): The length of each aligned audio snippet.

            pre_length (int): The number of samples to include before the threshold 
                              is first crossed.

            threshold (float): Used to find the start of the speech command. The 
                               speech command begins where the magnitude of the 
                               audio sample is greater than (threshold * max(samples)).

            envelope (bool): if True, use enveloping.

        Returns:
            aligned recording.
        \"\"\"
        if envelope:
            recording = utils.envelope(recording, 5400, 100)
        recording_threshold = threshold * np.max(recording)
        _i = pre_length
        while # YOUR CODE HERE
            _i = _i + 1
        snippet_start = min(_i - pre_length, len(recording) - length)
        snippet = # YOUR CODE HERE
        snippet_normalized = # YOUR CODE HERE
        return snippet_normalized
    """,
    name="_"
)


@app.cell
def _(align_recording, np):
    def align_data(data, length, pre_length, threshold, envelope=False):
        """
        align all audio samples in dataset. (apply align_recording to all rows of the 
        data matrix)

        Args:
            data (np.ndarray): Matrix where each row corresponds to a recording's 
                               audio samples.

            length (int): The length of each aligned audio snippet.

            pre_length (int): The number of samples to include before the threshold 
                              is first crossed.

            threshold (float): Used to find the start of the speech command. The 
                               speech command begins where the magnitude of the 
                               audio sample is greater than (threshold * max(samples)).

        Returns:
            Matrix of aligned recordings.
        """
        assert isinstance(data, np.ndarray) and len(data.shape) == 2, "'data' must be a 2D matrix"
        assert isinstance(length, int) and length > 0, "'length' of snippet must be an integer greater than 0"
        assert 0 <= threshold <= 1, "'threshold' must be between 0 and 1"
        snippets = []

        # Iterate over the rows in data
        for recording in data:
            snippets.append(align_recording(recording, length, pre_length, threshold, envelope))

        return np.vstack(snippets)
    return (align_data,)


@app.cell
def _(align_data, plt, selected_words_arr):
    def process_data(dict_raw, length, pre_length, threshold, plot=True, envelope=False):
        """
        Process the raw data given parameters and return it. (wrapper function for align_data)

        Args:
            dict_raw (np.ndarray): dictionary of all words: data matrix.
            length (int): The length of each aligned audio snippet.
            pre_length (int): The number of samples to include before the threshold is first crossed.
            threshold (float): Used to find the start of the speech command. The speech command begins where the
                               magnitude of the audio sample is greater than (threshold * max(samples)).
            plot (boolean): Plot the dataset if true.

        Returns:
            Processed data dictionary.
        """
        processed_dict = {}
        _word_number = 0
        for (key, word_raw) in dict_raw.items():
            word_processed = align_data(word_raw, length, pre_length, threshold, envelope=envelope)
            processed_dict[key] = word_processed
            if plot:
                plt.plot(word_processed.T)
                plt.title('Samples for "{}"'.format(selected_words_arr[_word_number]))
                _word_number = _word_number + 1
                plt.show()
        return processed_dict
    return (process_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ###Select Parameters

        Play around with different values for length, pre-length, and threshold for the data from the spoken word "Jack". Notice how the allignment of the samples change for different parameter values. The goal is to have the samples alligned (all the lines of the graphs are similar). Can you tell what word is being spoken?
        """
    )
    return


@app.cell(hide_code=True)
def _(length, mo, pre_length, threshold):
    mo.hstack([length, pre_length, threshold])
    return


@app.cell
def _(jack_data, length, pre_length, process_data, threshold):
    process_data(jack_data, length.value, pre_length.value, threshold.value, envelope=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, see how the values you chose effect the other words.""")
    return


@app.cell
def _(length, pre_length, process_data, threshold, train_dict):
    processed_train_dict = process_data(train_dict, length.value, pre_length.value, threshold.value, envelope=True)
    return (processed_train_dict,)


@app.cell(hide_code=True)
def _(mo, parameter_answer):
    mo.md(f"What values did you chose for length, pre-length, and threshold? What made this combination the best set of parameters?<br><br>{mo.as_html(parameter_answer)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='task3'></a>
        ## <span style="color:navy">Task 2: PCA via SVD</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### SVD/PCA Resources
        - http://www.ams.org/publicoutreach/feature-column/fcarc-svd
        - https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues
        - https://towardsdatascience.com/pca-and-svd-explained-with-numpy-5d13b0d2a4d8
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Task 2a: Generate and Preprocess PCA Matrix

        Now that we have our aligned data, we will build the PCA input matrix from that data by stacking all the data vertically.
        """
    )
    return


@app.cell
def _(np, processed_train_dict):
    processed_A = np.vstack(list(processed_train_dict.values()))
    return (processed_A,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The first step of PCA is to center the data's mean at zero and store it in demeaned_A. Please note that you want to get the mean of each feature (what are the features?). The function np.mean might be helpful here, along with specifying the axis parameter.""")
    return


@app.cell
def _(np, processed_A):
    # Zero-mean the matrix A
    mean_vec = np.mean(processed_A, axis=0)
    demeaned_A = processed_A - mean_vec
    print(processed_A.shape)
    print(mean_vec.shape)
    return demeaned_A, mean_vec


@app.cell
def _(mo, svd_hint):
    mo.md(
        f"<h3>Task 2b: PCA</h3><div>Take the SVD of your demeaned data<br>{svd_hint}</div>"
    )
    return


@app.cell
def _(demeaned_A, np):
    # Take the SVD of matrix demeaned_A
    U, S, Vt = np.linalg.svd(demeaned_A)
    return S, U, Vt


@app.cell(hide_code=True)
def _(mo, stem_hint):
    mo.md(f"<div>Visually inspect your sigma values. They should tell you how many principal components you need.<br>{stem_hint}</div>")
    return


@app.cell
def _():
    # Plot out the sigma values
    # Don't forgot to call plt.show() after you graph!
    # YOUR CODE HERE
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Task 3c: Choosing a Basis using Principal Components""")
    return


@app.cell
def _():
    return


@app.cell
def _(mo, proj_eq, proj_hint_1, proj_hint_2):
    mo.md(
       f'''Set the `new_basis` argument to be a basis of the first **PCA_comp_num** principal components.

                Sanity check: When you plot `new_basis` you should see a number of line plots equal to the number of principal components you've chosen. You should have 3 differently colored components, each of length ‘length’ that you chose above.
                **NOTE: The projection onto a subspace with orthogonal basis vectors (i.e. a matrix with orthogonal columns) reduces to an inner product:**

                {proj_eq}

                refer to eecs16b's [Spring 2024 note 13](https://eecs16b.org/notes/sp24/note13.pdf) for a proof of this.

                {proj_hint_1}
                {proj_hint_2}
                '''
            )
    return


@app.cell
def _(PCA_comp_num_slider, Vt, plt):
    # Plot the principal component(s)
    PCA_comp_num = PCA_comp_num_slider.value # 3 for now
    new_basis = Vt[:PCA_comp_num].T # Use PCA_comp_num
    plt.plot(new_basis)
    plt.show()
    return PCA_comp_num, new_basis


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now project the data in the matrix A onto the new basis and plot it. For three principal components, in addition to the 3D plot, we also provided 2D plots which correspond to the top and side views of the 3D plot. Do you see clustering? Do you think you can separate the data easily?""")
    return


@app.cell(hide_code=True)
def _(
    Axes3D,
    all_words_arr,
    cm,
    demeaned_A,
    new_basis,
    np,
    num_samples_train,
    plt,
):
    proj = np.dot(demeaned_A, new_basis)
    if new_basis.shape[1] == 3:
        _fig = plt.figure(figsize=(10, 5))
        _ax = _fig.add_subplot(111, projection='3d')
        for _i in range(len(all_words_arr)):
            Axes3D.scatter(_ax, *proj[_i * num_samples_train:num_samples_train * (_i + 1)].T, c=cm[_i], marker='o', s=20)
        plt.legend(all_words_arr, loc='center left', bbox_to_anchor=(1.07, 0.5))
        (_fig, _axs) = plt.subplots(1, 3, figsize=(15, 5))
        for _i in range(len(all_words_arr)):
            _axs[0].scatter(proj[_i * num_samples_train:num_samples_train * (_i + 1), 0], proj[_i * num_samples_train:num_samples_train * (_i + 1), 1], c=cm[_i], edgecolor='none')
            _axs[1].scatter(proj[_i * num_samples_train:num_samples_train * (_i + 1), 0], proj[_i * num_samples_train:num_samples_train * (_i + 1), 2], c=cm[_i], edgecolor='none')
            _axs[2].scatter(proj[_i * num_samples_train:num_samples_train * (_i + 1), 1], proj[_i * num_samples_train:num_samples_train * (_i + 1), 2], c=cm[_i], edgecolor='none')
        _axs[0].set_title('View 1')
        _axs[1].set_title('View 2')
        _axs[2].set_title('View 3')
        plt.legend(all_words_arr, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    elif new_basis.shape[1] == 2:
        _fig = plt.figure(figsize=(10, 5))
        for _i in range(len(all_words_arr)):
            plt.scatter(proj[_i * num_samples_train:num_samples_train * (_i + 1), 0], proj[_i * num_samples_train:num_samples_train * (_i + 1), 1], edgecolor='none')
        plt.legend(all_words_arr, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    return (proj,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Like in many AI applications, the data above are noisy, so we expect some classification errors. The important part is that you see strong clustering of your words. 

        If you don't see clustering, try to think about why this might be the case. Things you might want to ask yourself:
        - How does PCA create the clusters? 
        - Which characteristics of your waveform will PCA favor when clustering? 
        - How can you choose your words to maximize the differences between the classes?

        Once you think you have decent clustering, move on to automating classification. **Choose 4 out of the 6 words which form the most distinct clusters. You will be using these four words for the rest of this lab.**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='task4'></a>
        ## <span style="color:navy">Task 3: Clustering Data Points</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Implement `find_centroids`, which finds the center of each cluster.""")
    return


@app.cell
def _():
    def find_centroid(clustered_data):
        """Find the center of each cluster by taking the mean of all points in a cluster.
        It may be helpful to recall how you constructed the data matrix (e.g. which rows correspond to which word)

        Parameters:
            clustered_data (np.array): the data already projected onto the new basis (for one word)

        Returns: 
            centroids (list): The centroids of the clusters
        """

        return # YOUR CODE HERE
    return (find_centroid,)


@app.cell
def _(all_words_arr, find_centroid, num_samples_train, proj):
    centroids = []
    for _i in range(len(all_words_arr)):
        _centroid = find_centroid(proj[_i * num_samples_train:(_i + 1) * num_samples_train])
        centroids.append(_centroid)
    print(centroids)
    return (centroids,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Run the cell below to plot your centroids along with your projected data.""")
    return


@app.cell(hide_code=True)
def _(
    Axes3D,
    all_words_arr,
    centroids,
    cm,
    new_basis,
    np,
    num_samples_train,
    plt,
    proj,
    selected_words_arr,
):
    centroid_list = np.vstack(centroids)
    colors = cm[:len(centroids)]
    for (_i, _centroid) in enumerate(centroid_list):
        print('Centroid {} is at: {}'.format(_i, str(_centroid)))
    if new_basis.shape[1] == 3:
        _fig = plt.figure(figsize=(10, 7))
        _ax = _fig.add_subplot(111, projection='3d')
        for _i in range(len(selected_words_arr)):
            Axes3D.scatter(_ax, *proj[_i * num_samples_train:num_samples_train * (_i + 1)].T, c=cm[_i], marker='o', s=20)
        plt.legend(selected_words_arr, loc='center left', bbox_to_anchor=(1.07, 0.5))
        for _i in range(len(selected_words_arr)):
            Axes3D.scatter(_ax, *np.array([centroids[_i]]).T, c=cm[_i], marker='*', s=300)
        plt.title('Training Data')
        (_fig, _axs) = plt.subplots(1, 3, figsize=(15, 5))
        for _i in range(len(all_words_arr)):
            _axs[0].scatter(proj[_i * num_samples_train:num_samples_train * (_i + 1), 0], proj[_i * num_samples_train:num_samples_train * (_i + 1), 1], c=cm[_i], edgecolor='none')
            _axs[1].scatter(proj[_i * num_samples_train:num_samples_train * (_i + 1), 0], proj[_i * num_samples_train:num_samples_train * (_i + 1), 2], c=cm[_i], edgecolor='none')
            _axs[2].scatter(proj[_i * num_samples_train:num_samples_train * (_i + 1), 1], proj[_i * num_samples_train:num_samples_train * (_i + 1), 2], c=cm[_i], edgecolor='none')
        _axs[0].set_title('View 1')
        _axs[1].set_title('View 2')
        _axs[2].set_title('View 3')
        plt.legend(all_words_arr, loc='center left', bbox_to_anchor=(1, 0.5))
        _axs[0].scatter(centroid_list[:, 0], centroid_list[:, 1], c=colors, marker='*', s=300)
        _axs[1].scatter(centroid_list[:, 0], centroid_list[:, 2], c=colors, marker='*', s=300)
        _axs[2].scatter(centroid_list[:, 1], centroid_list[:, 2], c=colors, marker='*', s=300)
        plt.show()
    elif new_basis.shape[1] == 2:
        _fig = plt.figure(figsize=(10, 7))
        for _i in range(len(all_words_arr)):
            plt.scatter(proj[_i * num_samples_train:num_samples_train * (_i + 1), 0], proj[_i * num_samples_train:num_samples_train * (_i + 1), 1], c=colors[_i], edgecolor='none')
        plt.scatter(centroid_list[:, 0], centroid_list[:, 1], c=colors, marker='*', s=300)
        plt.legend(all_words_arr, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Training Data')
        plt.show()
    return centroid_list, colors


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='task3'></a>
        ## <span style="color:navy">Task 4: Testing your Classifier</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Great! Now that we have the means (centroid) for each word, let's evaluate performance. Recall that we will classify each data point according to the centroid with the least Euclidian distance to it.

        Before we perform classification, we need to do the same preprocessing to the test data that we did to the training data (enveloping, demeaning, projecting onto the PCA basis). You have already written most of the code for this part. However, note the difference in variable names as we are now working with test data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""First let's look at what our raw test data looks like.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Task 4a: Test Data Preprocessing""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's repeat what we did for our training data, so we can try out the test data on our trained model!""")
    return


@app.cell
def _(all_words_arr, plt, test_dict):
    _word_number = 0
    for _word_raw_test in test_dict.values():
        plt.plot(_word_raw_test.T)
        plt.title('Test sample for "{}"'.format(all_words_arr[_word_number]))
        _word_number = _word_number + 1
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Perform enveloping and trimming of our test data.""")
    return


@app.cell
def _(length, pre_length, process_data, test_dict, threshold):
    processed_test_dict = process_data(test_dict, length.value, pre_length.value, threshold.value, plot=True, envelope=True)
    return (processed_test_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Construct the PCA matrix. Refer to the code we used above for the training set!""")
    return


app._unparsable_cell(
    r"""
    processed_A_test = # YOUR CODE HERE
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Now we will do something slightly different.**
        Previously, you projected data onto your PCA basis with $(x - \bar{x})P$, where $\bar{x}$ is the mean vector, $x$ is a single row of `processed_A`, and $P$ is `new_basis`.
        We can rewrite this operation as:
        $$(x - \bar{x})P = xP - \bar{x}P = xP - \bar{x}_{\text{proj}}$$
        $$\bar{x}_{\text{proj}} = \bar{x}P$$
        Why might we want to do this? In the real world, we want our data to take up as little storage as possible. Instead of storing a length $n$ vector $\bar{x}$, we can precompute $\bar{x}_{\text{proj}} \in \mathbb{R}^3$ and store that instead!
        Compute $\bar{x}_{\text{proj}}$ using the **same mean vector** as the one computed with the training data.
        """
    )
    return


app._unparsable_cell(
    r"""
    projected_mean_vec = # YOUR CODE HERE
    print(new_basis.shape)
    print(mean_vec)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Project the test data onto the **same PCA basis** as the one computed with the training data.""")
    return


app._unparsable_cell(
    r"""
    projected_A_test = # YOUR CODE HERE
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Zero-mean the projected test data using the **`projected_mean_vec`**.""")
    return


@app.cell
def _(projected_A_test, projected_mean_vec):
    proj_1 = projected_A_test - projected_mean_vec
    print(projected_mean_vec.shape)
    return (proj_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plot the projections to see how well your test data clusters in this new basis. This will give you an idea of test classification accuracy.""")
    return


@app.cell
def _(
    Axes3D,
    all_words_arr,
    centroid_list,
    centroids,
    cm,
    colors,
    new_basis,
    np,
    num_samples_test,
    plt,
    proj_1,
):
    if new_basis.shape[1] == 3:
        _fig = plt.figure(figsize=(10, 7))
        _ax = _fig.add_subplot(111, projection='3d')
        for _i in range(len(all_words_arr)):
            Axes3D.scatter(_ax, *proj_1[_i * num_samples_test:num_samples_test * (_i + 1)].T, c=cm[_i], marker='o', s=20)
        plt.legend(all_words_arr, loc='center left', bbox_to_anchor=(1.07, 0.5))
        plt.title('Test Data')
        for _i in range(len(all_words_arr)):
            Axes3D.scatter(_ax, *np.array([centroids[_i]]).T, c=cm[_i], marker='*', s=300)
        (_fig, _axs) = plt.subplots(1, 3, figsize=(15, 5))
        for _i in range(len(all_words_arr)):
            _axs[0].scatter(proj_1[_i * num_samples_test:num_samples_test * (_i + 1), 0], proj_1[_i * num_samples_test:num_samples_test * (_i + 1), 1], c=cm[_i], edgecolor='none')
            _axs[1].scatter(proj_1[_i * num_samples_test:num_samples_test * (_i + 1), 0], proj_1[_i * num_samples_test:num_samples_test * (_i + 1), 2], c=cm[_i], edgecolor='none')
            _axs[2].scatter(proj_1[_i * num_samples_test:num_samples_test * (_i + 1), 1], proj_1[_i * num_samples_test:num_samples_test * (_i + 1), 2], c=cm[_i], edgecolor='none')
        _axs[0].set_title('View 1')
        _axs[1].set_title('View 2')
        _axs[2].set_title('View 3')
        plt.legend(all_words_arr, loc='center left', bbox_to_anchor=(1, 0.5))
        _axs[0].scatter(centroid_list[:, 0], centroid_list[:, 1], c=colors, marker='*', s=300)
        _axs[1].scatter(centroid_list[:, 0], centroid_list[:, 2], c=colors, marker='*', s=300)
        _axs[2].scatter(centroid_list[:, 1], centroid_list[:, 2], c=colors, marker='*', s=300)
        plt.show()
    elif new_basis.shape[1] == 2:
        _fig = plt.figure(figsize=(10, 7))
        for _i in range(len(all_words_arr)):
            plt.scatter(proj_1[_i * num_samples_test:num_samples_test * (_i + 1), 0], proj_1[_i * num_samples_test:num_samples_test * (_i + 1), 1], c=colors[_i], edgecolor='none')
        plt.scatter(centroid_list[:, 0], centroid_list[:, 1], c=colors, marker='*', s=300)
        plt.legend(all_words_arr, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Test Data')
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now that we have some idea of how our test data looks in our PCA basis, let's see how our data actually performs. Implement the `classify` function which takes in a data point after enveloping is applied and returns which word number it belongs to depending on the closed centroid in Euclidian distance.""")
    return


app._unparsable_cell(
    r"""
    def classify(word_set, data_point, new_basis, projected_mean_vec, centroids):
        \"\"\"Classifies a new voice recording into a word.

        Args:
            word_set (list): set of words that we've chosen
            data_point (np.array): new data point vector before demeaning and projection
            new_basis (np.array): the new processed basis to project on
            projected_mean_vec (np.array): the same projected_mean_vec as before
        Returns:
            (string): The classified word
        Hint:
            Remember to use 'projected_mean_vec'!
            np.argmin(), and np.linalg.norm() may also help!
        \"\"\"
        # TODO: classify the demeaned data point by comparing its distance to the centroids
        projected_data_point = # YOUR CODE HERE

        # hint: demean with the precomputed mean vector
        demeaned = # YOUR CODE HERE 

        # hint: we're returning the word that this data point classified as
        return # YOUR CODE HERE
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Try out the classification function below.""")
    return


@app.cell
def _(
    all_words_arr,
    centroids,
    classify,
    new_basis,
    processed_A_test,
    projected_mean_vec,
):
    # Try out the classification function
    # Modify the row index of processed_A_test to use other vectors
    print(classify(all_words_arr, processed_A_test[0,:], new_basis, projected_mean_vec, centroids))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Our goal is 80% accuracy for each word.** Apply the `classify` function to each sample and compute the accuracy for each word. If you do not meet 80% accuracy for each word, try to find different combinations of words/parameters that result in more distinct clusters in the plots of the projected data.""")
    return


@app.cell
def _(
    all_words_arr,
    centroids,
    classify,
    new_basis,
    np,
    num_samples_test,
    processed_A_test,
    projected_mean_vec,
):
    correct_counts = np.zeros(len(all_words_arr))
    for (row_num, data) in enumerate(processed_A_test):
        word_num = row_num // num_samples_test
        if classify(all_words_arr, data, new_basis, projected_mean_vec, centroids) == all_words_arr[word_num]:
            correct_counts[word_num - 1] = correct_counts[word_num - 1] + 1
    for _i in range(len(correct_counts)):
        print('Percent correct of word {} = {}%'.format(all_words_arr[_i], 100 * correct_counts[_i] / num_samples_test))
    return correct_counts, data, row_num, word_num


@app.cell(hide_code=True)
def _(PCA_comp_num_slider, mo):
    mo.md(f"""
    Why did we chose to classify with 3 PCA components? What would happen if we go down to 2 or even 1 PCA component? What would happen if we increase the number of PCA components? Use the slider below to try out different numbers of PCA components. Changing the slider automatically reruns the dependent cells above and you may see the precentages above change as a result. Give Marmio a couple seconds to rerun and render the new results.

    {mo.as_html(PCA_comp_num_slider)}
    """)
    return


@app.cell(hide_code=True)
def _(PCA_comp_num_ans, mo):
    mo.md(f"""
    What number of PCA components did you chose and why?

    {mo.as_html(PCA_comp_num_ans)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## <span style="color:green">CHECKOFF</span>

        ### When you are ready to get checked off, fill out the **[Checkoff Google Form](https://docs.google.com/forms/d/e/1FAIpQLSfIOjvEJXew-M0-h9uJ3C25UOdmmABFK0GGNl3o9p7po7Cc0A/viewform?usp=sf_link)**


        - **Have all questions, code, and plots completed in this notebook.** Your TA will check all your PCA code and plots.
        - **Show your GSI that you've achieved 80% accuracy on your test data for all 4 words.** 
        - **Be prepared to answer conceptual questions about the lab.**
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(csv, display, mo, train_dict, utils):
    # contains the variables for Marimo interative features
    # DO NOT MODIFY

    length = mo.ui.radio(
        options={"1000": 1000, "2000": 2000, "3000": 3000, "4000": 4000},
        value="1000",
        label="Length",
    )
    pre_length = mo.ui.radio(
        options={"100": 100, "200": 200, "300": 300, "400": 400},
        value="100",
        label="Pre-Length",
    )
    threshold = mo.ui.radio(
        options={"0.25": 0.25, "0.50": 0.5, "0.75": 0.75, "1.00": 1.00},
        value="0.25",
        label="Threshold",
    )
    jack_data = {"jack": train_dict["jack"]}

    record_button = mo.ui.run_button(label="Start Recording")
    live_classify_button = mo.ui.run_button(label="Start Recording")

    parameter_answer = mo.ui.text_area(placeholder="type your answer here ...")

    svd_hint = mo.as_html(
        mo.accordion({"Click for a hint": "np.linalg.svd may be useful here"}, lazy=False)
    )

    stem_hint = mo.as_html(
        mo.accordion({"Click for a hint": "use plt.stem to create a stem plot"}, lazy=False)
    )

    proj_hint_1 = mo.as_html(
        mo.accordion({"Hint #1": "Of the three outputs from the SVD function call, which one will contain the principal components onto which we want to project our data points? Do we need to transpose it?"}, lazy=False)
    )
    proj_hint_2 = mo.as_html(
        mo.accordion({"Hint #2": "For $A \\in \\mathbb{R}^{n \\times n}$, $U$ contains the basis of the column space of $A$, and $V$ contains the basis of the row space of $A$."}, lazy=False)
    )

    proj_eq = "$$\\text{proj}_{\\text{Col(Q)}} (\\vec{y}) = QQ^T \\vec{y} = \\sum^n_{i=1} \\langle \\vec{y}, \\vec{q}_i \\rangle \\vec{q}_i$$"

    PCA_comp_num_slider = mo.ui.slider(start=1, stop=10, value=3, label="Number of PCA Components")

    PCA_comp_num_ans = mo.ui.text_area(placeholder="type your answer here ...")

    def create_recording_csv(rate=5400, chunk=1024, record_seconds=3, num_recordings=50):
        filename = input("Enter the CSV file name (it will be created): ")
        if not filename.endswith('.csv'):
            print("Warning: filename does not end with .csv. Appending .csv to the filename.")
            filename = filename + '.csv'

        recording_count = 0

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            try:
                while recording_count < num_recordings:
                    user_input = input("Press Enter to start recording, or type 'd' and then Enter to remove the last entry, or type 'stop' and then Enter to stop recording: ")

                    if user_input == '':
                        # Record audio
                        audio_data = utils.record_audio(seconds=record_seconds, rate=rate, chunk=chunk)
                        writer.writerow(audio_data)
                        file.flush()

                        recording_count += 1

                        display.clear_output()
                        print(f"Audio data saved to CSV. Recorded {recording_count}/{num_recordings}. Length of vector: {len(audio_data)}\n")

                    elif user_input.lower() == 'd': 
                        if recording_count > 0:
                            # Remove the last entry
                            with open(filename, mode='r') as f:
                                lines = f.readlines()
                            with open(filename, mode='w') as f:
                                f.writelines(lines[:-1])

                            recording_count -= 1
                            display.clear_output()
                            print(f"Last recording removed. Recorded {recording_count}/{num_recordings}.")
                        else:
                            print("No entries to delete.")
                    elif user_input.lower() == 'stop': 
                        break

            except KeyboardInterrupt:
                print("Data collection stopped.")

        print("Data collection stopped. Saved to " + filename)
    return (
        PCA_comp_num_ans,
        PCA_comp_num_slider,
        create_recording_csv,
        jack_data,
        length,
        live_classify_button,
        parameter_answer,
        pre_length,
        proj_eq,
        proj_hint_1,
        proj_hint_2,
        record_button,
        stem_hint,
        svd_hint,
        threshold,
    )


if __name__ == "__main__":
    app.run()
