def get_input():
	user_input = ""
	user_input = input("csv file: ")

	return user_input

def read_file():
	tweets = []
	with open("kamputhaw-aug04-firecebu.csv", "r") as ins:
		for line in ins:
			separated = line.split(';')
			tweet = separated[4]
			tweets.append(tweet)
			print(tweet)
	return tweets

def write_file(tweets):
	f = open("clean.txt", "w")
	for tweet in tweets:
		f.write(tweet + "\n")
	f.close

def main():
	# user_input = str(get_input())
	# print(user_input)
	tweets = read_file()
	write_file(tweets)

if __name__ == '__main__':
	main()