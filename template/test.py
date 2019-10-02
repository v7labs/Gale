from multiprocessing import Pool

def f(x):
    return x+2


if __name__ == "__main__":

    pool = Pool(2)

    pool.map(f, [1,2,3])

    pool.close()

    print("done!")



