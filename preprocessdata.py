import os
from shutil import copyfile

"""
For more information about the dataset, visit the project website:

  https://github.com/switchablenorms/CelebAMask-HQ
"""


def main():
    # Mapping identities from the CelebA-HQ-identity.txt file
    identities = {}

    with open("./CelebA-HQ-identity.txt") as f:
        lines = f.readlines()
        for line in lines:
            file_name, identity = line.strip().split()
            identities[file_name] = identity

    # Displaying information about the dataset
    print(f"There are {len(set(identities.values()))} identities.")
    print(f"There are {len(identities.keys())} images.")

    # Setting up source and target directories
    source_root = "/mnt/ssd/projects/lorenzom/data/CelebAMask-HQ/CelebA-HQ-img/"
    target_root = "./identity_dataset/"

    # Creating target directories based on identities
    file_list = os.listdir(source_root)

    for file in file_list:
        identity = identities[file]
        source = os.path.join(source_root, file)
        target = os.path.join(target_root, str(identity), file)

        # Creating identity folders if they don't exist
        if not os.path.exists(os.path.join(target_root, str(identity))):
            os.makedirs(os.path.join(target_root, str(identity)))

        # Copying images to respective identity folders
        copyfile(source, target)

    # Processing and organizing dataset for training and testing
    folder_root = "./identity_dataset/"
    folder_list = os.listdir(folder_root)

    threshold = 15
    identity_cnt = 0
    train_images = 0
    test_images = 0
    train_ratio = 0.8

    for folder in folder_list:
        file_list = os.path.join(folder_root, folder)
        file_list = os.listdir(file_list)

        if len(file_list) >= threshold:
            identity_cnt += 1
            num_train = int(train_ratio * len(file_list))

            # Moving images to train folder
            for file in file_list[:num_train]:
                train_images += 1
                source = os.path.join(folder_root, folder, file)
                target = os.path.join(folder_root, "train", folder, file)

                if not os.path.exists(os.path.join(folder_root, "train", folder)):
                    os.makedirs(os.path.join(folder_root, "train", folder))

                os.rename(source, target)

            # Moving images to test folder
            for file in file_list[num_train:]:
                test_images += 1
                source = os.path.join(folder_root, folder, file)
                target = os.path.join(folder_root, "test", folder, file)

                if not os.path.exists(os.path.join(folder_root, "test", folder)):
                    os.makedirs(os.path.join(folder_root, "test", folder))

                os.rename(source, target)

    # Displaying final dataset statistics
    print(
        f"There are {identity_cnt} identities that have more than {threshold} images."
    )
    print(f"There are {train_images} train images.")
    print(f"There are {test_images} test images.")

    # Renaming final directory structure
    os.rename("./identity_dataset/train", "./facial_identity_dataset/train")
    os.rename("./identity_dataset/test", "./facial_identity_dataset/test")


if __name__ == "__main__":
    main()
