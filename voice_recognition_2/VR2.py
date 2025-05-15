import marimo

__generated_with = "0.11.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #Voice Recognition Lab 2: Extending SVD/PCA Classification with Spectral Analysis

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
        * [Task 1: Data Preprocessing](#task1)
        * [Task 2: Spectral Analysis](#task2)
        * [Task 3: Data Reshaping](#task3)
        * [Task 4: PCA via SVD](#task4)
        * [Task 5: Testing your Classifier](#task5)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Before you start, have the notebook from last week handy, as you might be copying code from it to this notebook.""")
    return


@app.cell
def _():
    import wave
    import numpy as np
    import csv
    from tqdm.notebook import trange
    from IPython import display
    import time
    import matplotlib.pyplot as plt
    import scipy.io
    import utils
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.signal import stft, spectrogram
    # '%matplotlib inline' command supported automatically in marimo

    cm = ['blue', 'red', 'green', 'orange', 'black', 'purple']
    return (
        Axes3D,
        cm,
        csv,
        display,
        np,
        plt,
        scipy,
        spectrogram,
        stft,
        time,
        trange,
        utils,
        wave,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='task1'></a>
        # <span style="color:navy"> Task 1: Data Preprocessing
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We will repeat the data preprocessing step from lab 1 below.""")
    return


@app.cell
def _():
    all_words_arr = ['jack', 'jason', 'jessica', 'principalcomponent']
    return (all_words_arr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's begin by splitting our data into a training and testing set with a 70/30 split. Run the code below to do so.""")
    return


@app.cell
def _(all_words_arr, np, utils):
    train_test_split_ratio = 0.7
    train_dict = {}
    test_dict = {}
    for _i in range(len(all_words_arr)):
        word_raw = utils.read_csv('{}.csv'.format(all_words_arr[_i]))
        (word_raw_train, word_raw_test) = utils.train_test_split(word_raw, train_test_split_ratio)
        train_dict[all_words_arr[_i]] = word_raw_train
        test_dict[all_words_arr[_i]] = word_raw_test
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
        word_raw_test,
        word_raw_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Align the recordings as we did last week. Paste in the `align_recording` function you wrote last week.""")
    return


@app.cell
def align_recording():
    def align_recording(recording, length, pre_length, threshold, envelope=False):
        """
        align a single audio samples according to the given parameters.

        Args:
            recording (np.ndarray): a single audio sample.
            length (int): The length of each aligned audio snippet.
            pre_length (int): The number of samples to include before the threshold is first crossed.
            threshold (float): Used to find the start of the speech command. The speech command begins where the
                magnitude of the audio sample is greater than (threshold * max(samples)).
            envelope (bool): if True, use enveloping.

        Returns:
            aligned recording.
        """

        # TODO: PASTE IN YOUR ALIGN_RECORDING FUNCTION FROM LAST WEEK
    return (align_recording,)


@app.cell
def _(align_recording, np):
    def align_data(data, length, pre_length, threshold, envelope=False):
        """
        align all audio samples in dataset. (apply align_recording to all rows of the data matrix)

        Args:
            data (np.ndarray): Matrix where each row corresponds to a recording's audio samples.
            length (int): The length of each aligned audio snippet.
            pre_length (int): The number of samples to include before the threshold is first crossed.
            threshold (float): Used to find the start of the speech command. The speech command begins where the
                magnitude of the audio sample is greater than (threshold * max(samples)).

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
def _(align_data, plt):
    def process_data(selected_words_arr, dict_raw, length, pre_length, threshold, plot=True, envelope=False):
        """
        Process the raw data given parameters and return it. (wrapper function for align_data)

        Args:
            dict_raw (np.ndarray): Raw data collected.
            data (np.ndarray): Matrix where each row corresponds to a recording's audio samples.
            length (int): The length of each aligned audio snippet.
            pre_length (int): The number of samples to include before the threshold is first crossed.
            threshold (float): Used to find the start of the speech command. The speech command begins where the
                magnitude of the audio sample is greater than (threshold * max(samples)).
            plot (boolean): Plot the dataset if true.

        Returns:
            Processed data dictionary.
        """
        processed_dict = {}
        word_number = 0
        for (key, word_raw) in dict_raw.items():
            word_processed = align_data(word_raw, length, pre_length, threshold, envelope=envelope)
            processed_dict[key] = word_processed
            if plot:
                plt.plot(word_processed.T)
                plt.title('Samples for "{}"'.format(selected_words_arr[word_number]))
                word_number = word_number + 1
                plt.show()
        return processed_dict
    return (process_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Align your recordings. **NOTE: we want to set `envelope=False` for spectral analysis!**""")
    return


@app.cell
def _(all_words_arr, process_data, train_dict):
    # TODO: Edit the parameters to get the best alignment.
    length = ... # YOUR CODE HERE
    pre_length = 400 # Modify this as necessary
    threshold = ... # YOUR CODE HERE

    # align training and test data
    processed_train_dict = process_data(all_words_arr, train_dict, length, pre_length, threshold, envelope=False)
    return length, pre_length, processed_train_dict, threshold


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='task2'></a>
        # <span style="color:navy">Task 2: Spectral Analysis</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You have seen spectrograms previously in the Shazam lab. As a reminder, it's a DFT calculated on many small snippets of the signal in question. The spectrogram is a 2d plot that gives insights on both temporal and frequency information in the signal.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## <span style="color:navy"> Utilizing Spectrograms to extract features""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        By obtaining the spectrogram, we are able to capture the unique time-varying frequency footprint of the signal. After collecting these coefficients, we will be able to use PCA, a method of low-rank approximation, to convert the spectrogram coefficients into a basis that maximizes the amount of variance.

        Calculating the spectrogram first will allow PCA to identify variations in both frequency and temporal details. 

        Generate a spectrogram for each recording below.
        """
    )
    return


@app.cell
def _(np):
    def spectrogram_single_recording(data, sample_rate=5400, return_f_t=False):
        """
        calculate spectrogram of one recording.

        Args:
            data (np.array): single recording (row)
            sample_rate (int): sampling rate in Hz
            return_f_t (bool): indicate whether to only return Zxx or f,t,Zxx (for plotting)
        Returns:
            f (np.array): frequency index array
            t (np.array): time index array
            Zxx (np.array): array of 2D arrays (spectrogram result of each row)
        """
        (f, t, Zxx) = ...
        if return_f_t:
            return (f, t, Zxx)
        else:
            return Zxx

    def spectrogram_recordings(values, sample_rate=5400, return_f_t=False):
        """
        calculate spectrogram of multiple recordings.

        Args:
            values (np.array): recordings matrix
            sample_rate (int): sampling rate in Hz
            return_f_t (bool): indicate whether to only return Zxx or f,t,Zxx (for plotting)
        Returns:
            f (np.array): frequency index array
            t (np.array): time index array
            Zxx (np.array): array of 2D arrays (spectrogram result of each row)
        """
        temp_vals = []
        for (word, recordings) in values:
            for _i in range(recordings.shape[0]):
                (f, t, Zxx) = spectrogram_single_recording(recordings[_i, :], sample_rate=sample_rate, return_f_t=True)
                ...
        if return_f_t:
            return (f, t, np.array(temp_vals))
        else:
            return np.array(temp_vals)
    return spectrogram_recordings, spectrogram_single_recording


@app.cell
def _(processed_train_dict, spectrogram_recordings):
    f, t, spectrogram_results = spectrogram_recordings(processed_train_dict.items(), return_f_t=True)
    return f, spectrogram_results, t


@app.cell
def _(f, np, plt, spectrogram_results, t):
    # show the first spectrogram result
    plt.pcolormesh(t, f, np.abs(spectrogram_results[0]), shading='gouraud')
    plt.title('Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## <span style="color:navy"> Mel-scaled Spectrograms

        The Mel-scaling is a frequency scaling takes advantage of knowledge of the human auditory system. Rather than representing frequencies linearly, it uses this prior knowledge to compress higher frequencies and expand lower frequencies, similar to our ears. In other words, it reforms the frequency scaling to be more similar to the way humans perceive pitch. This has the potential to extract features that are more relevant to human auditory perception.

        In DFT (Discrete Fourier Transform), the frequency bins are dictated by the DFT frequency bins. In the Mel-scaled equivalent, the frequency bins use a new rescaled unit called 'Mels', which is what the `n_mels` argument for the function below denotes. Thus, the returned spectrogram result would be a 2D array of shape (# timesteps x # Mels) that contains the spectrogram value in each location.

        We will provide the functions needed for generating Mel-scaled spectrograms, so you can treat it as a black box if you'd like.
        """
    )
    return


@app.cell
def _(np, spectrogram_recordings, spectrogram_single_recording, utils):
    # run mel-scaled spectrogram on single recording
    def mel_spectrogram_single_recording(data, sample_rate=5400, n_fft=256, n_mels=100, return_f_t=False):
        mel_filter = utils.mel_filter_bank(sample_rate, n_fft, n_mels)
        f, t, spectrogram_result = spectrogram_single_recording(data, return_f_t=True)
        mel_spectrogram_result = np.array(utils.apply_mel_filter([spectrogram_result], mel_filter))[0]
        mel_freqs = utils.mel_frequencies(n_mels, sample_rate, n_fft)
        if return_f_t:
            return mel_freqs, t, mel_spectrogram_result
        else:
            return mel_spectrogram_result

    # run mel-scaled spectrogram on entire dataset
    def mel_spectrogram_recordings(vals, sample_rate=5400, n_fft=256, n_mels=100, return_f_t=False):
        mel_filter = utils.mel_filter_bank(sample_rate, n_fft, n_mels)
        f, t, spectrogram_results = spectrogram_recordings(vals, return_f_t=True)
        mel_spectrogram_results = np.array(utils.apply_mel_filter(spectrogram_results, mel_filter))
        mel_freqs = utils.mel_frequencies(n_mels, sample_rate, n_fft)
        if return_f_t:
            return mel_freqs, t, mel_spectrogram_results
        else:
            return mel_spectrogram_results
    return mel_spectrogram_recordings, mel_spectrogram_single_recording


@app.cell
def _(mel_spectrogram_recordings, processed_train_dict):
    (mel_freqs, t_1, mel_spectrogram_results) = mel_spectrogram_recordings(processed_train_dict.items(), return_f_t=True)
    return mel_freqs, mel_spectrogram_results, t_1


@app.cell
def _(mel_freqs, mel_spectrogram_results, plt, t_1):
    plt.pcolormesh(t_1, mel_freqs, mel_spectrogram_results[0], shading='gouraud')
    plt.title('Mel Spectrogram')
    plt.ylabel('Frequency [Mels]')
    plt.xlabel('Time [sec]')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='task3'></a>
        # <span style="color:navy"> Task 3: Data Reshaping</span>


        Currently, our spectrogram matrices (both regular and Mel-scaled) are 2d (time and frequency). This doesn't lend itself well to PCA, which requires a single vector per measurement. 

        Therefore, we must flatten our array to a 1d format. There are many ways to accomplish this--we will be exploring and attempting some of the options below.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Sanity check**: What do each axis in our spectrogram_results array represent?""")
    return


@app.cell
def _(spectrogram_results):
    spectrogram_results.shape
    return


@app.cell
def _(mel_spectrogram_results):
    mel_spectrogram_results.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## <span style="color:navy"> Flattening

        In flattening, we are simply taking all the elements from the matrix and place them into a single sequence, thereby having no data loss. Keeping in mind that the two dimensions are time and frequency, we might choose to flatten row-wise, or column-wise. We may even decide to use a different approach, zig-zagging through the matrix to flatten it. For now, we will use row-wise flattening, but we encourage you to experiment with approaches in the later part of the lab.
        """
    )
    return


@app.cell
def _(mo):
    mo.image("public/sc1.png")
    return


@app.cell(hide_code=True)
def _(hint_1, mo):
    mo.md(f"""
    Hint for apply_flattening_single_recording(data):
    {mo.as_html(hint_1)}
    """)
    return


@app.cell
def apply_flattening_single_recording():
    # flatten the matrix row-wise.
    def apply_flattening_single_recording(data):
        return ... # YOUR CODE HERE
    return (apply_flattening_single_recording,)


@app.cell
def apply_flattening():
    # flatten the matrix row-wise.
    def apply_flattening(vals):
        return ... # YOUR CODE HERE
    return (apply_flattening,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## <span style="color:navy"> Aggregation of standard deviation / variance features over frames

        Another method to convert our matrix to a vector is to use mean and variances over frequency bins for each time frame. This does lose some information, but very specific frequency information is likely redundant for a simple voice classification scheme. Generally, the memory tradeoff for this feature is worth it.
        """
    )
    return


@app.cell
def _(mo):
    mo.image("public/sc2.png")
    return


@app.cell(hide_code=True)
def _(hint_2, hint_3, mo):
    mo.md(f"""
    Hints for apply_aggregation_single_recording().
    {mo.as_html(hint_2)}
    {mo.as_html(hint_3)}
    """)
    return


@app.cell
def apply_aggregation_single_recording():
    # TODO: Compute mean and variance over frequency bins for each time frame

    # apply aggregation for one recording
    def apply_aggregation_single_recording(data):
        mean_feature, variance_feature = ... # YOUR CODE HERE 

        # Stack features to get a combined feature set
        return ... # YOUR CODE HERE
    return (apply_aggregation_single_recording,)


app._unparsable_cell(
    r"""

    # apply aggregation for multiple recordingsdef apply_aggregation(vals):
        mean_features, variance_features = ... # YOUR CODE HERE (same as above, but you might have to specify a different axis argument)

        # Stack features to get a combined feature set
        return ... # YOUR CODE HERE (same as above, but you might have to specify a different axis argument)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## <span style="color:navy"> Computing all 4 possible configurations for spectral analysis + reshaping""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We have two methods of spectrogram generation (Mel-scaled spectrograms and regular spectrograms) and two methods of data aggregation (regular flattening vs mean & variance aggregation). That gives us 4 possible combinations, so let's generate a processed dataset for all 4 cases so we can compare!""")
    return


@app.cell
def _():
    # use the functions we wrote above!

    # TODO: mel scaled spectrogram + flattening
    processed_A_mel_flattening = ... # YOUR CODE HERE
    print(f"processed_A_mel_flattening shape: {processed_A_mel_flattening.shape}")

    # TODO: regular spectrogram + flattening
    processed_A_spectrogram_flattening = ... # YOUR CODE HERE
    print(f"processed_A_spectrogram_flattening shape: {processed_A_spectrogram_flattening.shape}")

    # TODO: mel scaled spectrogram + aggregation
    processed_A_mel_aggregated = ... # YOUR CODE HERE
    print(f"processed_A_mel_aggregated shape: {processed_A_mel_aggregated.shape}")

    # TODO: regular spectrogram + aggregation
    processed_A_spectrogram_aggregated = ... # YOUR CODE HERE
    print(f"processed_A_spectrogram_aggregated shape: {processed_A_spectrogram_aggregated.shape}")
    return (
        processed_A_mel_aggregated,
        processed_A_mel_flattening,
        processed_A_spectrogram_aggregated,
        processed_A_spectrogram_flattening,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='task4'></a>
        # <span style="color:navy">Task 4: PCA via SVD</span>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now we will be repeating our PCA steps from last week's lab. Refer to the code you've written to complete this section. In the empty `processed_A`, choose one of the four processed datasets from above. Rerun this section for different cases to compare the clustering characteristics and results!""")
    return


@app.cell
def _(Axes3D, all_words_arr, cm, np, num_samples_train, plt):
    processed_A = ...
    mean_vec = ...
    demeaned_A = ...
    (U, S, Vt) = ...
    plt.figure()
    plt.stem(S)
    plt.title('Stem Plot of Sigma Values')
    new_basis = ...
    plt.figure()
    plt.plot(new_basis)
    plt.title('New Basis Vectors')
    _proj = ...
    centroids = []
    for _i in range(len(all_words_arr)):
        _centroid = np.mean(_proj[_i * num_samples_train:(_i + 1) * num_samples_train], axis=0)
        centroids.append(_centroid)
    _centroid_list = np.vstack(centroids)
    _colors = cm[:len(centroids)]
    if new_basis.shape[1] == 3:
        _fig = plt.figure(figsize=(10, 7))
        _ax = _fig.add_subplot(111, projection='3d')
        for _i in range(len(all_words_arr)):
            Axes3D.scatter(_ax, *_proj[_i * num_samples_train:num_samples_train * (_i + 1)].T, c=cm[_i], marker='o', s=20)
        plt.legend(all_words_arr, loc='center left', bbox_to_anchor=(1.07, 0.5))
        for _i in range(len(all_words_arr)):
            Axes3D.scatter(_ax, *np.array([centroids[_i]]).T, c=cm[_i], marker='*', s=300)
        plt.title('Training Data')
        (_fig, _axs) = plt.subplots(1, 3, figsize=(15, 5))
        for _i in range(len(all_words_arr)):
            _axs[0].scatter(_proj[_i * num_samples_train:num_samples_train * (_i + 1), 0], _proj[_i * num_samples_train:num_samples_train * (_i + 1), 1], c=cm[_i], edgecolor='none')
            _axs[1].scatter(_proj[_i * num_samples_train:num_samples_train * (_i + 1), 0], _proj[_i * num_samples_train:num_samples_train * (_i + 1), 2], c=cm[_i], edgecolor='none')
            _axs[2].scatter(_proj[_i * num_samples_train:num_samples_train * (_i + 1), 1], _proj[_i * num_samples_train:num_samples_train * (_i + 1), 2], c=cm[_i], edgecolor='none')
        _axs[0].set_title('View 1')
        _axs[1].set_title('View 2')
        _axs[2].set_title('View 3')
        plt.legend(all_words_arr, loc='center left', bbox_to_anchor=(1, 0.5))
        _axs[0].scatter(_centroid_list[:, 0], _centroid_list[:, 1], c=_colors, marker='*', s=300)
        _axs[1].scatter(_centroid_list[:, 0], _centroid_list[:, 2], c=_colors, marker='*', s=300)
        _axs[2].scatter(_centroid_list[:, 1], _centroid_list[:, 2], c=_colors, marker='*', s=300)
    elif new_basis.shape[1] == 2:
        _fig = plt.figure(figsize=(10, 7))
        for _i in range(len(all_words_arr)):
            plt.scatter(_proj[_i * num_samples_train:num_samples_train * (_i + 1), 0], _proj[_i * num_samples_train:num_samples_train * (_i + 1), 1], c=_colors[_i], edgecolor='none')
        plt.scatter(_centroid_list[:, 0], _centroid_list[:, 1], c=_colors, marker='*', s=300)
        plt.legend(all_words_arr, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Training Data')
    plt.show()
    for (_i, _centroid) in enumerate(_centroid_list):
        print('Centroid {} is at: {}'.format(_i, str(_centroid)))
    return S, U, Vt, centroids, demeaned_A, mean_vec, new_basis, processed_A


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <a id='task5'></a>
        # <span style="color:navy"> Task 5: Testing your Classifier</span>
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
    mo.md(r"""## <span style="color:navy"> Test Data Preprocessing""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Perform enveloping and trimming of our test data.""")
    return


@app.cell
def _(
    all_words_arr,
    length,
    pre_length,
    process_data,
    test_dict,
    threshold,
):
    processed_test_dict = process_data(all_words_arr, test_dict, length, pre_length, threshold, envelope=False)
    return (processed_test_dict,)


@app.cell
def _(
    mel_spectrogram_recordings,
    processed_test_dict,
    spectrogram_recordings,
):
    # run spectrogram and mel spectrogram on the test set as well
    spectrogram_results_test = spectrogram_recordings(processed_test_dict.items(), return_f_t=False)
    mel_spectrogram_results_test = mel_spectrogram_recordings(processed_test_dict.items(), return_f_t=False)

    # generate the four possible configurations for the test data as well

    # TODO: mel scaled spectrogram + flattening
    processed_A_mel_flattening_test = ... # YOUR CODE HERE
    print(f"processed_A_mel_flattening_test shape: {processed_A_mel_flattening_test.shape}")

    # TODO: regular spectrogram + flattening
    processed_A_spectrogram_flattening_test = ... # YOUR CODE HERE
    print(f"processed_A_spectrogram_flattening_test shape: {processed_A_spectrogram_flattening_test.shape}")

    # TODO: mel scaled spectrogram + aggregation
    processed_A_mel_aggregated_test = ... # YOUR CODE HERE
    print(f"processed_A_mel_aggregated_test shape: {processed_A_mel_aggregated_test.shape}")

    # TODO: regular spectrogram + aggregation
    processed_A_spectrogram_aggregated_test = ... # YOUR CODE HERE
    print(f"processed_A_spectrogram_aggregated_test shape: {processed_A_spectrogram_aggregated_test.shape}")
    return (
        mel_spectrogram_results_test,
        processed_A_mel_aggregated_test,
        processed_A_mel_flattening_test,
        processed_A_spectrogram_aggregated_test,
        processed_A_spectrogram_flattening_test,
        spectrogram_results_test,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now we will project our processed test dataset the same way we did as before. As a reminder, we precomputed the mean vector $\bar{x}_{\text{proj}}$ to save storage in our test classification and live classification:

        $$(x - \bar{x})P = xP - \bar{x}P = xP - \bar{x}_{\text{proj}} \\ \bar{x}_{\text{proj}} = \bar{x}P$$
        """
    )
    return


@app.cell
def _(Axes3D, all_words_arr, cm, new_basis, np, num_samples_test, plt):
    processed_A_test = ...
    projected_mean_vec = ...
    _proj = ...
    centroids_1 = []
    for _i in range(len(all_words_arr)):
        _centroid = np.mean(_proj[_i * num_samples_test:(_i + 1) * num_samples_test], axis=0)
        centroids_1.append(_centroid)
    _centroid_list = np.vstack(centroids_1)
    _colors = cm[:len(centroids_1)]
    if new_basis.shape[1] == 3:
        _fig = plt.figure(figsize=(10, 7))
        _ax = _fig.add_subplot(111, projection='3d')
        for _i in range(len(all_words_arr)):
            Axes3D.scatter(_ax, *_proj[_i * num_samples_test:num_samples_test * (_i + 1)].T, c=cm[_i], marker='o', s=20)
        plt.legend(all_words_arr, loc='center left', bbox_to_anchor=(1.07, 0.5))
        for _i in range(len(all_words_arr)):
            Axes3D.scatter(_ax, *np.array([centroids_1[_i]]).T, c=cm[_i], marker='*', s=300)
        plt.title('Training Data')
        (_fig, _axs) = plt.subplots(1, 3, figsize=(15, 5))
        for _i in range(len(all_words_arr)):
            _axs[0].scatter(_proj[_i * num_samples_test:num_samples_test * (_i + 1), 0], _proj[_i * num_samples_test:num_samples_test * (_i + 1), 1], c=cm[_i], edgecolor='none')
            _axs[1].scatter(_proj[_i * num_samples_test:num_samples_test * (_i + 1), 0], _proj[_i * num_samples_test:num_samples_test * (_i + 1), 2], c=cm[_i], edgecolor='none')
            _axs[2].scatter(_proj[_i * num_samples_test:num_samples_test * (_i + 1), 1], _proj[_i * num_samples_test:num_samples_test * (_i + 1), 2], c=cm[_i], edgecolor='none')
        _axs[0].set_title('View 1')
        _axs[1].set_title('View 2')
        _axs[2].set_title('View 3')
        plt.legend(all_words_arr, loc='center left', bbox_to_anchor=(1, 0.5))
        _axs[0].scatter(_centroid_list[:, 0], _centroid_list[:, 1], c=_colors, marker='*', s=300)
        _axs[1].scatter(_centroid_list[:, 0], _centroid_list[:, 2], c=_colors, marker='*', s=300)
        _axs[2].scatter(_centroid_list[:, 1], _centroid_list[:, 2], c=_colors, marker='*', s=300)
    elif new_basis.shape[1] == 2:
        _fig = plt.figure(figsize=(10, 7))
        for _i in range(len(all_words_arr)):
            plt.scatter(_proj[_i * num_samples_test:num_samples_test * (_i + 1), 0], _proj[_i * num_samples_test:num_samples_test * (_i + 1), 1], c=_colors[_i], edgecolor='none')
        plt.scatter(_centroid_list[:, 0], _centroid_list[:, 1], c=_colors, marker='*', s=300)
        plt.legend(all_words_arr, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('Training Data')
    plt.show()
    for (_i, _centroid) in enumerate(_centroid_list):
        print('Centroid {} is at: {}'.format(_i, str(_centroid)))
    return centroids_1, processed_A_test, projected_mean_vec


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Implement the classify function.""")
    return


@app.cell(hide_code=True)
def _(hint_4, mo):
    mo.md(f"""
    Hint for classify()
    {mo.as_html(hint_4)}
    """)
    return


@app.cell
def _(all_words_arr):
    def classify(data_point, new_basis, projected_mean_vec, centroids):
        """Classifies a new voice recording into a word.

        Args:
            data_point: new data point vector before demeaning and projection
            new_basis: the new processed basis to project on
            projected_mean_vec: the same projected_mean_vec as before
        Returns:
            Word number (should be in {1, 2, 3, 4} -> you might need to offset your indexing!)

        """
        # TODO: classify the demeaned data point by comparing its distance to the centroids
        projected_data_point = ... # YOUR CODE HERE
        demeaned = ... # YOUR CODE HERE
        return all_words_arr[...] # YOUR CODE HERE
    return (classify,)


@app.cell
def _(
    centroids_1,
    classify,
    new_basis,
    processed_A_test,
    projected_mean_vec,
):
    print(classify(processed_A_test[0, :], new_basis, projected_mean_vec, centroids_1))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    hint_1 = mo.accordion({'Hint 1': 'np.reshape may be helpful here.'})
    hint_2 = mo.accordion({'Hint 2': 'use np.mean and np.var'}) 
    hint_3 = mo.accordion({'Hint 3': 'use np.concatenate'})
    hint_4 = mo.accordion({'Hint 4':'Remember to use projected_mean_vec!np.argmin(), and np.linalg.norm() may also help!'})
    return hint_1, hint_2, hint_3, hint_4


if __name__ == "__main__":
    app.run()
