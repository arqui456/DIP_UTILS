import tools as tools
import cv2

def main():
	while True:
		if cv2.waitKey(0) == ord('q'):
			return 
		print("Enter which function do you want to use: ")
		inp = int(input())
		print("enter the img path: ")
		if inp == 1:
			tools.readGrayScaleImg(input())
		elif inp == 2:
			tools.loadColor(input())
		elif inp == 3:
			tools.loadColorWithChannels(input())
		elif inp == 4:
			tools.histogramGray(input())
		elif inp == 5:
			tools.histogramColor(input())
		elif inp == 6:
			tools.createImg(int(input()))
		elif inp == 7:
			tools.addScalar(input())
		elif inp == 8:
			tools.mergeImages(input().split(" "))
		elif inp == 9:
			tools.imageMax(input().split(" "))
		elif inp == 10:
			tools.imageAbs(input().split(" "))
		elif inp == 11:
			tools.videoDiff()
		elif inp == 12:
			tools.addNoise(input())
		elif inp == 13:
			tools.addSaltAndPepper(input())
		elif inp == 14:
			tools.setOP(input().split(" "))

	cv2.waitKey(0)

if __name__ == "__main__":
	main()
