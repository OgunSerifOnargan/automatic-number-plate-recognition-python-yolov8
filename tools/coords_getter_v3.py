import pygame
import cv2
import os
import numpy as np

def adjust_window_size(screen_width, screen_height,img_width, img_height, menu_width):
    img_scale_ratio = img_width/img_height
    window_width = screen_width/2
    window_height = window_width / img_scale_ratio

    window_width += menu_width
    return window_width, window_height
    
def take_photo(recorder_option):
    max_attempts = 30
    cap = cv2.VideoCapture(recorder_option)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Waiting for a non-black frame...")

    for _ in range(max_attempts):
        ret, frame = cap.read()
    temp_img_path = r"temp.jpg"
    cap.release()
    cv2.imwrite(temp_img_path, frame)
    image = pygame.image.load(temp_img_path)
    os.remove(temp_img_path)
    return image

def first_point_remover(cords):
    cords.pop()
    first_point = None
    return cords, first_point

def second_point_remover(cords, lines):
    cords.pop()
    lines.pop()
    second_point = None

    return cords, lines, second_point


def get_coords(recorder_option, lineCounter=False, circle_drawer=False, obstacleChecker=False):
#region initialize items
    cords = []
    lines = []  # Store lines (pairs of points)
    white = (255, 255, 255)
    green = (0, 255, 0)
    gray = (128, 128, 128)
    black = (0, 0, 0)

    pygame.init()
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
#endregion

#region gets
#get image
    if lineCounter or obstacleChecker:
        image = take_photo(recorder_option)
    if circle_drawer:
        image = pygame.image.load(recorder_option)
    
# Get the screen size
    screen_size = pygame.display.Info()
    screen_width, screen_height = screen_size.current_w, screen_size.current_h

#Get the image size
    img_width = image.get_width()
    img_height = image.get_height()
    #Get the menu and buttons sizes
    menu_width = 150
    window_width, window_height = adjust_window_size(screen_width, screen_height,img_width, img_height, menu_width) 
#set items' sizes
    gray_area_rect = pygame.Rect(window_width - menu_width, 0, menu_width, window_height)
    button_width, button_height = 130, 30
    button_rects = [
        pygame.Rect(window_width - button_width - 10, 10, button_width, button_height),
        pygame.Rect(window_width - button_width - 10, 50, button_width, button_height),
    ]
#endregion

#screen construction
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Line drawer")
    pygame.draw.rect(screen, gray, gray_area_rect)


#region initialize variables
    first_point = None
    second_point = None
    success = True
#endregion

    while success:
            mouse_x, mouse_y = pygame.mouse.get_pos()

            for event in pygame.event.get():
                #display Buttons
                for i, button_rect in enumerate(button_rects):
                    pygame.draw.rect(screen, white, button_rect)
                    pygame.draw.rect(screen, black, button_rect, 2)

                    if i == 0:
                        text = font.render("Undo", True, black)
                    elif i == 1:
                        text = font.render("Done", True, black)
                    text_rect = text.get_rect(center=button_rect.center)
                    screen.blit(text, text_rect.topleft)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    #MENU BUTTON EVENTS
                    for i, button_rect in enumerate(button_rects):
                        if button_rect.collidepoint(mouse_x, mouse_y):
                            #UNDO
                            if i == 0: 
                                if first_point is None and second_point is None and len(lines) > 0:
                                    cords, lines, second_point = second_point_remover(cords, lines)
                                    first_point = cords[-1]
                                elif second_point is None and len(cords)>0:
                                    cords, first_point = first_point_remover(cords)
                                elif second_point is not None:
                                    cords, lines, second_point = second_point_remover(cords, lines)
                                elif first_point is not None and second_point is None:
                                    cords, first_point = first_point_remover(cords)
                                else:
                                    pass

                            #DONE
                            elif i == 1:
                                success = False
                                break


                    if 0 <= mouse_x < window_width - menu_width and 0 <= mouse_y < window_height:
                        if first_point is None:
                            first_point = (mouse_x, mouse_y)
                            cords.append(first_point)
                            if lineCounter == True:
                                continue

                        elif second_point is None:
                            second_point = (mouse_x, mouse_y)
                            cords.append(second_point)
                            lines.append([first_point, second_point])
                            if lineCounter == True:
                                continue

                        if second_point is not None:
                            first_point = (mouse_x, mouse_y)
                            cords.append(first_point)
                            second_point = None

            # Scale the image to fit the screen
            scaled_image = pygame.transform.scale(image, (window_width - menu_width, window_height))
            screen.blit(scaled_image, (0, 0))


            for line in lines:
                pygame.draw.line(screen, green, line[0], line[1], 4)

            for cord in cords:
                pygame.draw.circle(screen, green, cord, 4)

            if first_point and not second_point:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if 0 <= mouse_x < window_width - menu_width and 0 <= mouse_y < window_height:
                    pygame.draw.line(screen, green, first_point, (mouse_x, mouse_y), 4)

            pygame.display.flip()

            clock.tick(60)
    cv2.destroyAllWindows()

    if lineCounter:
        lines_return = []
        for i, line in enumerate(lines):
            x1 = line[0][0]
            x1_return = int(x1/(window_width - menu_width) * img_width)
            y1 = line[0][1]
            y1_return = int(y1/(window_height) * img_height)
            x2 = line[1][0]
            x2_return = int(x2/(window_width - menu_width) * img_width)
            y2 = line[1][1]
            y2_return = int(y2/(window_height) * img_height)
            lines_return.append([[x1_return, y1_return],[x2_return, y2_return]])
        cv2.destroyAllWindows()
        return lines_return
        
    if circle_drawer or obstacleChecker:
        points_return = []
        for i, line in enumerate(lines):
            x1 = line[0][0]
            x1_return = int(x1/(window_width - menu_width) * img_width)
            y1 = line[0][1]
            y1_return = int(y1/(window_height) * img_height)
            x2 = line[1][0]
            x2_return = int(x2/(window_width - menu_width) * img_width)
            y2 = line[1][1]
            y2_return = int(y2/(window_height) * img_height)
            line = [[x1_return, y1_return], [x2_return, y2_return]]

            if i == len(lines)-1:
                continue
            elif i == 0:
                points_return.append(line[0])
                points_return.append(line[1])
            else:
                points_return.append(line[1])

        points_return = np.array(points_return)
        print(points_return)
        cv2.destroyAllWindows()
        return points_return


# result = get_coords()
# print(result)


