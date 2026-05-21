def combine_scores(profile, sharp_norm, face_sharp, expo, noise, blur, subj, aesth):
    if profile == "sports":
        final = (
            0.40 * sharp_norm +
            0.10 * face_sharp +
            0.15 * expo +
            0.10 * noise +
            0.10 * blur +
            0.05 * subj +
            0.10 * aesth
        )
    elif profile == "burlesque":
        final = (
            0.20 * sharp_norm +
            0.20 * face_sharp +
            0.20 * expo +
            0.10 * noise +
            0.05 * blur +
            0.10 * subj +
            0.15 * aesth
        )
    else:  # derby
        final = (
            0.35 * sharp_norm +
            0.15 * face_sharp +
            0.15 * expo +
            0.10 * noise +
            0.10 * blur +
            0.05 * subj +
            0.10 * aesth
        )

    return final
