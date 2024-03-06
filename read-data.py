#%%
from astropy.io import fits
from astropy.table import Table

# Read the FITS file
fits_file = fits.open('selected-objects-small.fits')
catalog = Table(fits_file[1].data)
fits_file.close()

# Convert to pandas DataFrame 
catalog_df = catalog.to_pandas()

# show columsn
catalog_df.columns

#%%
import os

# Filter the DataFrame to include only rows for which an image exists
available_images = set(os.listdir('sdss-images'))  # Get all image filenames in the folder
catalog_df = catalog_df[catalog_df['SPECOBJID_1'].apply(lambda id: f"{id}.jpg" in available_images)]

#%%
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_images(df, folder):
    """ funcion to read images and target values """
    images = []
    oh_targets = []  # Metallicity targets
    sfr_targets = []  # Star Formation Rate targets
    logmass_targets = []  # Stellar Mass targets

    for _, row in df.iterrows():
        img_id = str(row['SPECOBJID_1'])
        oh_target = row['OH_P50']  # Corrected variable name
        sfr_target = row['SFR_TOT_P50']
        logmass_target = row['LGM_TOT_P50']
        img_path = os.path.join(folder, f"{img_id}.jpg")

        if os.path.exists(img_path):  # Check if the image exists
            img = load_img(img_path, target_size=(128, 128))
            img_array = img_to_array(img)
            images.append(img_array)
            oh_targets.append(oh_target)  # Use the corrected variable name
            sfr_targets.append(sfr_target)
            logmass_targets.append(logmass_target)

    return np.array(images), np.array(oh_targets), np.array(sfr_targets), np.array(logmass_targets)


#%%

import numpy as np
import os
from sklearn.model_selection import train_test_split

# Initial split: 60% training, 40% temporary (to be split into validation and test)
train_df, temp_df = train_test_split(catalog_df, test_size=0.4, random_state=42)

# Split the temporary set into 50% validation and 50% test, which is effectively 20% each of the original dataset
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


# Load and preprocess images and targets for each set
X_train, Y_train_oh, Y_train_sfr, Y_train_logmass = load_images(train_df, 'sdss-images')
X_val, Y_val_oh, Y_val_sfr, Y_val_logmass = load_images(val_df, 'sdss-images')
X_test, Y_test_oh, Y_test_sfr, Y_test_logmass = load_images(test_df, 'sdss-images')


# Normalize image data
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

#%%

# Example of visualisation of 6 random imagesin the test set

import matplotlib.pyplot as plt

# Generate 6 random indices
random_indices = np.random.choice(len(X_test), size=6, replace=False)

# Plot the images
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flatten()):
    index = random_indices[i]
    image = X_test[index]
    label = Y_test_oh[index]
    ax.imshow(image)
    ax.set_title(f"oh_p50: {label}")
    ax.axis('off')

plt.tight_layout()
plt.show()
