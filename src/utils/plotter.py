from matplotlib import pyplot as plt

def plot_loss(filename, title):
    with open(filename + ".txt", "r") as f:
        losses = []
        for line in f:
            if line == "":
                continue
            if len(line.split('.')) > 2:
                continue
            losses.append(float(line))
        f.close()
    plt.subplot(1,1,1)
    plt.plot(losses, color='blue', lw=2)
    # plt.yscale('log')
    plt.xlabel('Optimization step', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title(title, fontsize=26)
    plt.gcf().set_size_inches(16, 12)
    plt.savefig("loss.png")

def read_file(filename: str):
    try:
        ret = []
        with open(filename + ".txt", "r") as f:
            for line in f:
                if line == "" or line == "\n":
                    continue
                ret.append(int(line))
        return ret
    except IOError as identifier:
        return None

def plot_rate(title):
    rate = []
    index = 0
    while True:
        r = read_file(str(index))
        if r is not None:
            rate.extend(r)
            index += 1
        else:
            break

    mark_every = 100
    solved = []
    solved_in = []
    generation = []
    batch = []
    for gen in rate:
        if gen >= 0:
            batch.append(1)
            generation.append(gen)
        else:
            batch.append(0)
            generation.append(512)
        if len(batch) == mark_every:
            solved.append(sum(batch)/mark_every)
            solved_in.append(sum(generation)/mark_every)
            generation.clear()
            batch.clear()
    if len(batch)>0:
        print(sum(batch), len(batch))
        solved.append(sum(batch)/len(batch))
        solved_in.append(sum(generation)/len(generation))

    plt.subplots(3, 1)
    plt.subplot(2,1,1)
    plt.title(title, fontsize=26)
    # plt.ylim(0,1)
    plt.plot(solved, color='blue', lw=2)
    plt.xlabel('Per iteration', fontsize=18)
    plt.ylabel('Average solving rate', fontsize=18)

    plt.subplot(2,1,2)
    # plt.ylim(0,512)
    plt.plot(solved_in, color='red', lw=2)
    plt.xlabel('Per iteration', fontsize=18)
    plt.ylabel('Average generation', fontsize=18)
    plt.gcf().set_size_inches(16, 12)
    plt.savefig("rate.png")

loss_dir = None

def set_loss_directory(loss_directory: str):
    global loss_dir
    loss_dir = loss_directory
    # clear content
    open(loss_dir, "w").close()

def save_loss(loss_array: list):
    global loss_dir
    if not loss_array:
        return
    print("Average Loss:", sum(loss_array)/len(loss_array))
    loss_dir
    with open(loss_dir, "a") as f:
        f.write("\n".join([str(x) for x in loss_array])+"\n")
        f.flush()
    f.close()

if __name__ == "__main__":
    title = 'ARCHITECTURE DETAILS'
    plot_loss("loss", title)
    plot_rate(title)