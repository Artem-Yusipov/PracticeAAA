#include <iostream>
#include <opencv2/opencv.hpp>
#include <ViZDoom.h>


std::string path = "C:\\practice\\vizdoom";
auto game = std::make_unique<vizdoom::DoomGame>();
const unsigned int sleepTime = 1000 / vizdoom::DEFAULT_TICRATE;
auto screenBuff = cv::Mat(480, 640, CV_8UC3);

void RunTask1(int episodes)
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

    auto greyscale = cv::Mat(480, 640, CV_8UC1);

    std::vector<double> action;

    for (auto i = 0; i < episodes; i++)
    {
        game->newEpisode();
        std::cout << "Episode #" << i + 1 << std::endl;

        while (!game->isEpisodeFinished())
        {
            const auto& gamestate = game->getState();

            std::memcpy(screenBuff.data, gamestate->screenBuffer->data(), gamestate->screenBuffer->size());

            cv::extractChannel(screenBuff, greyscale, 2);

            cv::threshold(greyscale, greyscale, 180, 255, cv::THRESH_BINARY);

            cv::imshow("Output Window", greyscale);

            double reward = game->makeAction({ 0, 0, 1 });


            std::cout << reward << " ";
            cv::waitKey(sleepTime);
        }

        std::cout << std::endl << game->getTotalReward() << std::endl;
    }
}


int main()
{
    game->setViZDoomPath(path + "\\vizdoom.exe");
    game->setDoomGamePath(path + "\\freedoom2.wad");

    cv::namedWindow("Control Window", cv::WINDOW_AUTOSIZE);

    auto episodes = 10;


    RunTask1(episodes);


    game->close();
}
