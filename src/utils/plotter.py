from matplotlib import pyplot as plt

def plot_loss(filename, title):
    with open(filename + ".txt", "r") as f:
        losses = []
        for line in f:
            if line == "":
                continue
            loss = line.split(", ")[0]
            losses.append(float(loss))
        f.close()
    plt.subplot(1,1,1)
    plt.plot(losses, color='blue', lw=2)
    plt.yscale('log')
    plt.xlabel('Optimization step', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title(title, fontsize=26)
    plt.gcf().set_size_inches(16, 12)
    plt.savefig("loss.png")

def plot_rate(filename, title):
    with open(filename + ".txt", "r") as f:
        mark_every = 100
        solved = []
        solved_in = []
        generation = []
        batch = []
        for line in f:
            if line == "" or line == "\n":
                continue
            if int(line) > 0:
                batch.append(1)
                generation.append(int(line))
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
        f.close()

    plt.subplots(3, 1)
    plt.subplot(2,1,1)
    plt.title(title, fontsize=26)
    plt.ylim(0,1)
    plt.plot(solved, color='blue', lw=2)
    plt.xlabel('Per ' + str(mark_every) + " Problems", fontsize=18)
    plt.ylabel('Average solving rate', fontsize=18)

    plt.subplot(2,1,2)
    plt.ylim(0,512)
    plt.plot(solved_in, color='red', lw=2)
    plt.xlabel('Per ' + str(mark_every) + " Problems", fontsize=18)
    plt.ylabel('Average generation', fontsize=18)
    plt.gcf().set_size_inches(16, 12)
    plt.savefig("rate.png")

if __name__ == "__main__":
    title = 'Crossover 1HL 8N lr=10^-6'
    plot_loss("loss", title)
    plot_rate("rate", title)