# 88ClicksPerHour

## Requirements

* Java 8
* Scala >= 2.11.8
* sbt >= 0.13.16
* Spark >= 2.3.0

## Setup

After dowloading the repository (`git clone https://github.com/JohanBrunet/88ClicksPerHour.git`), place yourself inside the main folder: 
```
cd 88ClicksPerHour
```

:warning: Do not remove or modify the **model** folder

Run the `sbt compile` command to compile the project

## Usage

From there run the following command to run the program:
```
sbt run
```

You will be prompted with a message asking you to chose the path to the JSON file you want to test. \
Enter the absolute path to the file on your computer, then hit <kbd>Enter</kbd>

The program will then run on its own using the data provided by the file you specified and the model provided with this repository.

## Results

After the program is finished running you can find the results in the *.csv* file that is created inside the [predictions](predictions/) folder.