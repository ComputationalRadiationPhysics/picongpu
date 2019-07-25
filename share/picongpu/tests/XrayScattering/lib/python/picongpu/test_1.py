from checks import compare_with_fft
from checks import check_summation


def main():

    electrons_check = compare_with_fft('e', 1)
    ions_check = compare_with_fft('i', 28)
    summation_check = check_summation()
    if summation_check and electrons_check and ions_check:
        print("All tests passed.")
    else:
        print("Some tests didn't pass.")
        print("electrons test {})"
              "".format(electrons_check))
        print("ion test {}"
              "".format(ions_check))
        print("check summation test {}"
              "".format(summation_check))


if __name__ == '__main__':
    main()
