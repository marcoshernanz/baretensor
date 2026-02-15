import bt


def main() -> None:
    d = bt.Dog("Max")
    print(d.name)
    print(d.bark())
    d.name = "Charlie"
    print(d.bark())


if __name__ == "__main__":
    main()
