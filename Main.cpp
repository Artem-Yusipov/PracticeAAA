#include <iostream>
#include <opencv2/opencv.hpp>
#include <ViZDoom.h>


std::string path = "C:\\practice\\vizdoom";
auto game = std::make_unique<vizdoom::DoomGame>();
const unsigned int sleepTime = 1000 / vizdoom::DEFAULT_TICRATE;
auto episodes = 10;
auto screenBuff = cv::Mat(480, 640, CV_8UC3);
double count = 0;


void RunTask1(int episode)
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




void RunTask2(int episode)
{
	try
	{
		game->loadConfig(path + "\\scenarios\\task2.cfg");
		game->init();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	std::vector<double> actions[4];

	auto image = cv::Mat(480, 640, CV_8UC3);
	auto greyscale = cv::Mat(480, 640, CV_8UC1);

	cv::Mat clusters;

	actions[0] = { 1, 0, 0, 0 };
	actions[1] = { 0, 1, 0, 0 };
	actions[2] = { 0, 0, 1, 0 };
	actions[3] = { 0, 0, 0, 1 };

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
				game->makeAction(actions[3]);
			}

			cv::waitKey(sleepTime);
		}
		std::cout << game->getTotalReward() << std::endl;
		count += game->getTotalReward();
	}
}




void RunTask3(int episodes) {
	std::vector<double> actions[4];
	actions[0] = { 1, 0, 0, 0 };
	actions[1] = { 0, 1, 0, 0 };
	actions[2] = { 0, 0, 1, 0 };
	actions[3] = { 0, 0, 0, 1 };
	try
	{
		game->loadConfig(path + "/scenarios/task3.cfg");
		game->init();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	for (auto i = 0; i < episodes; i++)
	{
		game->newEpisode();
		std::cout << "Episode #" << i + 1 << std::endl;

		while (!game->isEpisodeFinished())
		{
			const auto& gamestate = game->getState();

			std::memcpy(screenBuff.data, gamestate->screenBuffer->data(), gamestate->screenBuffer->size());

			cv::Mat img = screenBuff;
			cv::Mat med = cv::imread("./sprites/Pickups/media0.png");
			cv::Mat result;

			double minval, maxval; cv::Point minLoc, maxLoc;
			int res_cols = img.cols - med.cols + 1;
			int res_rows = img.rows - med.rows + 1;

			result.create(res_cols, res_rows, CV_32FC1);
			cv::matchTemplate(img, med, result, 4);
			cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

			cv::Rect roi(290, 0, 60, result.rows);
            result = result(roi);
			cv::minMaxLoc(result, &minval, &maxval, &minLoc, &maxLoc);

			rectangle(img, maxLoc, cv::Point(maxLoc.x + med.cols, maxLoc.y + med.rows), cv::Scalar::all(0), 2, 8, 0);
			rectangle(result, minLoc, cv::Point(minLoc.x + med.cols, minLoc.y + med.rows), cv::Scalar::all(0), 2, 8, 0);

			if (maxLoc.y < 390) {
				if (290 + maxLoc.x < 300) game->makeAction(actions[0]);
				else if (290 + maxLoc.x > 340) game->makeAction(actions[1]);
				else game->makeAction(actions[3]);
			}
			else game->makeAction(actions[0]);
			cv::imshow("Gray", result);

			cv::waitKey(sleepTime);
		}

		std::cout << game->getTotalReward() << std::endl;
		count += game->getTotalReward(); 
	}
}



void RunTask4(int episodes) {
	std::vector<double> actions[4];
	actions[0] = { 1, 0, 0, 0 };
	actions[1] = { 0, 1, 0, 0 };
	actions[2] = { 0, 0, 1, 0 };
	actions[3] = { 0, 0, 0, 1 };
	try
	{
		game->loadConfig(path + "/scenarios/task4.cfg");
		game->init();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	for (auto i = 0; i < episodes; i++)
	{
		game->newEpisode();
		std::cout << "Episode #" << i + 1 << std::endl;

		while (!game->isEpisodeFinished())
		{
			const auto& gamestate = game->getState();

			std::memcpy(screenBuff.data, gamestate->screenBuffer->data(), gamestate->screenBuffer->size());

			cv::Mat img = screenBuff;
			cv::Mat med = cv::imread("./sprites/Pickups/media0.png");
			cv::Mat result;

			double minval, maxval; cv::Point minLoc, maxLoc;
			int res_cols = img.cols - med.cols + 1;
			int res_rows = img.rows - med.rows + 1;

			result.create(res_cols, res_rows, CV_32FC1);
			cv::matchTemplate(img, med, result, 4);
			cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

			cv::Rect roi(290, 0, 60, result.rows);
			result = result(roi);
			cv::minMaxLoc(result, &minval, &maxval, &minLoc, &maxLoc);

			rectangle(img, maxLoc, cv::Point(maxLoc.x + med.cols, maxLoc.y + med.rows), cv::Scalar::all(0), 2, 8, 0);
			rectangle(result, minLoc, cv::Point(minLoc.x + med.cols, minLoc.y + med.rows), cv::Scalar::all(0), 2, 8, 0);

			if (maxLoc.y < 390) {
				if (290 + maxLoc.x < 300) game->makeAction(actions[0]);
				else if (290 + maxLoc.x > 340) game->makeAction(actions[1]);
				else game->makeAction(actions[3]);
			}
			else game->makeAction(actions[0]);
			cv::imshow("Gray", result);

			cv::waitKey(sleepTime);
		}

		std::cout << game->getTotalReward() << std::endl;
		count += game->getTotalReward();
	}
}



void RunTask5(int episodes) {
	std::vector<double> actions[3];
	actions[0] = { 0, 0, 0 };
	actions[1] = { 1, 0, 0, };
	actions[2] = { 0, 1, 0 };
	try
	{
		game->loadConfig(path + "/scenarios/task5.cfg");
		game->init();
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

	for (auto i = 0; i < episodes; i++)
	{
		game->newEpisode();
		std::cout << "Episode #" << i + 1 << std::endl;

		while (!game->isEpisodeFinished())
		{
			const auto& gamestate = game->getState();

			std::memcpy(screenBuff.data, gamestate->screenBuffer->data(), gamestate->screenBuffer->size());

			cv::Mat img = screenBuff;
			cv::Mat med = cv::imread("./sprites/Effects/bal1a0.png");
			cv::Mat result;

			double minval, maxval; cv::Point minLoc, maxLoc;
			int res_cols = img.cols - med.cols + 1;
			int res_rows = img.rows - med.rows + 1;

			result.create(res_cols, res_rows, CV_32FC1);
			cv::matchTemplate(img, med, result, 4);
			cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

			cv::minMaxLoc(result, &minval, &maxval, &minLoc, &maxLoc);

			rectangle(img, maxLoc, cv::Point(maxLoc.x + med.cols, maxLoc.y + med.rows), cv::Scalar::all(0), 2, 8, 0);
			rectangle(result, minLoc, cv::Point(minLoc.x + med.cols, minLoc.y + med.rows), cv::Scalar::all(0), 2, 8, 0);

			if (maxLoc.y < 390) {
				if (minLoc.x < 20) game->makeAction(actions[2]);
				if ( maxLoc.x < 320) game->makeAction(actions[1]);
				else if (maxLoc.x > 320) game->makeAction(actions[2]);
				else game->makeAction(actions[0]);
			}
			else game->makeAction(actions[0]);
			cv::imshow("Gray", result);

			cv::waitKey(sleepTime);
		}

		std::cout << game->getTotalReward() << std::endl;
		count += game->getTotalReward();
	}
}




int main()
{
    game->setViZDoomPath(path + "\\vizdoom.exe");
    game->setDoomGamePath(path + "\\freedoom2.wad");

    RunTask5(episodes);

    std::cout << " " << std::endl;
    std::cout << "Average sum " << count / 10 << std::endl;

    game->close();
}
