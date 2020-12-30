class Image:
    Name = ""
    Data = []


# READ Training and testing Data

# read file
def ReadFile(FileName, Resise = True):
    f = open(FileName, "r")
    images = []
    tmp = []
    for x in f:
        s = str(x).split(',')
        for i in range(len(s)):
            if i + 1 == len(s):
                newImage = Image()
                newImage.Name = s[i]
                newImage.Data = tmp
                images.append(newImage)
                tmp = []
            else:
                if Resise:
                    tmp.append(int(s[i]) / 16)  # para o interval stay in 0 e 1
                else:
                    tmp.append(float(s[i]))

    return images
