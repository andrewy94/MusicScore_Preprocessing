from controller.upload_controller import UploadController
from controller.omr_controller import OMRController
from view.upload_view import UploadView
from view.omr_view import OMRView

def main():
    upload_view = UploadView()
    upload_controller = UploadController(upload_view)
    image_path = upload_controller.select_image()
    
    if not image_path:
        print("No valid image. Exiting process.")
        return

    omr_view = OMRView()
    omr_controller = OMRController(omr_view)
    omr_controller.run(image_path)

if __name__ == "__main__":
    main()