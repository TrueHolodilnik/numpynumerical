
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

def main():

    alg = "default"  # advanced_williams/default
    data = pd.read_csv('btc_v_d.csv')
    delay_max = 14
    start_capital = 1000
    show_every_delay_plot = Tru
    show_macdsignal_plot = False
    show_williams_plot = True

    Williams_period = 28

    MAX_DATA_LENGTH = len(data.values)
    # if (MAX_DATA_LENGTH > 1000): MAX_DATA_LENGTH = 1000
    price_vec = [0] * MAX_DATA_LENGTH
    MACD_vec = [0] * MAX_DATA_LENGTH
    SIGNAL_vec = [0] * MAX_DATA_LENGTH
    Williams_vec = [0] * MAX_DATA_LENGTH

    for i in range(MAX_DATA_LENGTH): price_vec[i] = data.values[i][4]

    #
    # EMA calculation
    #
    def EMA(n, index, vec):
        amount = 0
        suma = 0
        for k in range(0, n):
            x = pow(1 - 2 / (n + 1), k)
            suma += x
            x *= vec[index - k]
            amount += x
        return amount / suma

    #
    # MACD calculation
    #
    def MACD(index, vec):

        if index >= 26:
            EMA26 = EMA(12, index, vec)
            EMA12 = EMA(26, index, vec)
            return EMA12 - EMA26
        else: return 0

    #
    # SIGNAL calculation
    #
    def SIGNAL(index, vec):
        if index >= 35: return EMA(9, index, vec)
        else: return 0

    #
    # Williams indicator calculation for advanced alg
    #
    def Williams(index, vec):
        if index - Williams_period > 0:
            highest = 0
            lowest = 999999999999999999999
            Rt = 0
            for i in range(index - Williams_period):
                if vec[i] < lowest: lowest = vec[i]
                if vec[i] > highest: highest = vec[i]
            if (highest - lowest) != 0:
                Rt = (vec[i] - highest)/(highest - lowest) * 100
            return Rt
        else: return 0


    for i in range(MAX_DATA_LENGTH): MACD_vec[i] = MACD(i, price_vec)
    for i in range(MAX_DATA_LENGTH): SIGNAL_vec[i] = SIGNAL(i, MACD_vec)
    for i in range(MAX_DATA_LENGTH): Williams_vec[i] = Williams(i, price_vec)

    #
    # MACD and SIGNAL plot
    #
    if show_macdsignal_plot:
        plt.plot(MACD_vec, label = "MACD")
        plt.plot(SIGNAL_vec, label = "SIGNAL")
        plt.plot(price_vec, label="Original cost")
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.xlabel('Time axis (days)')
        plt.ylabel('Values')
        plt.grid(True)
        plt.legend()
        plt.show()

    if show_williams_plot:
        temp = [0] * MAX_DATA_LENGTH
        for i in range(MAX_DATA_LENGTH): temp[i] = -80
        plt.plot(temp, label="Lower bound")
        for i in range(MAX_DATA_LENGTH): temp[i] = -20
        plt.plot(temp, label="Upper bound")
        plt.plot(Williams_vec, label="Williams")
        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.grid(True)
        plt.legend()
        plt.show()

    #
    # BUY/SELL simulation with different delays in days
    # one plot for every day
    #
    results = [0] * delay_max
    for delay in range(1,delay_max):
        if show_every_delay_plot:
            plt.plot(price_vec, label="INPUT", color="yellow")
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(18.5, 10.5)
        buy_sell = []
        if alg == "default":
            for i in range(delay,MAX_DATA_LENGTH):
                if MACD_vec[i - delay] >= SIGNAL_vec[i - delay] and MACD_vec[i] >= SIGNAL_vec[i]:
                    buy_sell.append("sell")
                elif MACD_vec[i - delay] <= SIGNAL_vec[i - delay] and MACD_vec[i] >= SIGNAL_vec[i]:
                    buy_sell.append("buy")
                else:
                    buy_sell.append(0)
        elif alg == "advanced_williams":
            for i in range(delay, MAX_DATA_LENGTH):
                if (MACD_vec[i - delay] >= SIGNAL_vec[i - delay] and MACD_vec[i] >= SIGNAL_vec[i]) or (Williams_vec[i] > -20):
                    buy_sell.append("sell")
                elif (MACD_vec[i - delay] <= SIGNAL_vec[i - delay] and MACD_vec[i] >= SIGNAL_vec[i]) or (Williams_vec[i] < -80):
                    buy_sell.append("buy")
                else:
                    buy_sell.append(0)

        capital = start_capital
        coins = 0
        for i in range(MAX_DATA_LENGTH - delay):
            if buy_sell[i] == "buy":
                bought = False
                while capital >= price_vec[i]:
                    capital -= price_vec[i]
                    coins += 1
                    bought = True
                if bought and show_every_delay_plot: plt.scatter(i, price_vec[i], c='green')

            elif buy_sell[i] == "sell":
                sold = False
                while coins > 1:
                    coins -= 1
                    capital += price_vec[i]
                    sold = True
                if sold and show_every_delay_plot: plt.scatter(i, price_vec[i], c='red')

        results[delay] = capital - start_capital
        if show_every_delay_plot:
            plt.title("Total profit: " + str(capital - start_capital) + ". Red dot - sell. Green dot - buy. Delay: " + str(delay) + " days.")
            plt.show()

    #
    # Profit/delay plot
    #
    results[0] = start_capital
    plt.plot(results)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.xlabel('Delay in days')
    plt.ylabel('Total capital')
    plt.title('Final capital/delay dependence')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
