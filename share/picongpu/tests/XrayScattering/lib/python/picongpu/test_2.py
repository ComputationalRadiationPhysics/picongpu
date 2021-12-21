from checks import compare_with_fft


def main():

    electrons_check = compare_with_fft('e', 1)
    if electrons_check:
        print("All tests passed.")
    else:
        print("Some tests didn't pass.")
        print("electrons test {}"
              "".format(electrons_check))


if __name__ == '__main__':
    main()
