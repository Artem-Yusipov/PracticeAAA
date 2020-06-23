#include <iostream>
#include <opencv2/opencv.hpp>
#include <ViZDoom.h>

double count = 0;
std::string path = "C:\\practice\\vizdoom";
auto game = std::make_unique<vizdoom::DoomGame>();
const unsigned int sleepTime = 1000 / vizdoom::DEFAULT_TICRATE;

void runTask1(int episode)
{
	try
	{
		game->loadConfig(path + "\\scenarios\\task1.cfg");
		game->init();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	std::vector<double> actions[3];

	auto image = cv::Mat(480, 640, CV_8UC3);
	auto greyscale = cv::Mat(480, 640, CV_8UC1);

	cv::Mat clusters;

	actions[0] = { 1,0,0 };
	actions[1] = { 0,1,0 };
	actions[2] = { 0,0,1 };

	for (auto i = 0; i < episode; i++)
	{
		game->newEpisode();
		std::cout << "Episode #" << i + 1 << std::endl;

		while (!game->isEpisodeFinished())
		{
			const auto& gameState = game->getState();
			std::memcpy(image.data, gameState->screenBuffer->data(), gameState->screenBuffer->size());

			std::vector<cv::Point2f> centers;
			std::vector<cv::Point2f> points(0);
			for (int x = 0; x < 640; x++)
			{
				for (int y = 0; y < 480; y++)
				{
					if (int(image.at<cv::Vec3b>(y, x)[2]) > 130 && int(image.at<cv::Vec3b>(y, x)[0]) < 50)
					{
						greyscale.at<unsigned char>(y, x) = 255;
						points.push_back(cv::Point2f(x, y));
					}
					else
					{
						greyscale.at<unsigned char>(y, x) = 0;
					}
				}
			}

			greyscale.convertTo(greyscale, CV_32F);

			cv::Mat samples = greyscale.reshape(1, greyscale.total());

			cv::kmeans(points, 1, clusters, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_RANDOM_CENTERS, centers);

			greyscale.convertTo(greyscale, CV_8UC3);

			for (int i = 0; i < centers.size(); i++)
			{
				cv::Point c = centers[i];

				cv::circle(image, c, 5, cv::Scalar(0, 0, 255), -1, 8);
				cv::rectangle(image, cv::Rect(c.x - 25, c.y - 25, 50, 50), cv::Scalar(0, 0, 255));
			}

			for (int i = 0; i < points.size(); i++)
			{
				cv::circle(image, points[i], 2, cv::Scalar(0, 255, 0));
			}

			imshow("Grey", greyscale);
			imshow("Claster", image);

			if ((centers[0].x - 320) > 35)
			{
				game->makeAction(actions[1]);
			}
			else if ((centers[0].x - 320) < -35)
			{
				game->makeAction(actions[0]);
			}
			else
			{
				game->makeAction(actions[2]);
			}

			cv::waitKey(sleepTime);
		}
		std::cout << game->getTotalReward() << std::endl;
		count += game->getTotalReward();
	}
}



int main()
{
	try
	{
		game->setViZDoomPath(path + "\\vizdoom.exe");
		game->setDoomGamePath(path + "\\freedoom2.wad");
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	auto episodes = 10;

	runTask1(episodes);

	std::cout << std::endl << "Avarage sum: " << count / episodes << std::endl;

	game->close();
}
