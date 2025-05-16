import matplotlib.pyplot as plt
import os
import fitz  # PyMuPDF
import argparse


    # # Load the PDF files
    # pdf1_path = 'outputs_main_results/model_performance_dotted_comparison_ty.pdf'
    # pdf2_path = '/Users/tianyu/Work/paper2code/model_paper_knowledge_debug_ty.pdf'

def combine_pdfs_side_by_side(pdf1_path, pdf2_path, output_path):
    """Combine two PDFs side by side directly using PyMuPDF, ensuring they have the same height"""
    print(f"Combining PDFs: {pdf1_path} and {pdf2_path}")
    
    try:
        # Open both PDFs
        doc1 = fitz.open(pdf1_path)  # Model performance (right side)
        doc2 = fitz.open(pdf2_path)  # Knowledge cutoff (left side)
        
        # Create new PDF for output
        output = fitz.open()
        
        # Use the shorter of the two if they differ in length
        num_pages = min(len(doc1), len(doc2))
        
        for i in range(num_pages):
            page1 = doc1[i]
            page2 = doc2[i]
            
            # Get original page sizes
            rect1 = page1.rect
            rect2 = page2.rect
            
            # We'll make both PDFs the same height
            target_height = max(rect1.height, rect2.height)
            
            # Calculate scaling factors to maintain aspect ratio
            scale1 = target_height / rect1.height
            scale2 = target_height / rect2.height
            
            # Calculate new widths after scaling
            new_width1 = rect1.width * scale1
            new_width2 = rect2.width * scale2
            
            # Create a new page in the output document with the combined width
            # Add a small amount of overlap to eliminate any line between the PDFs
            overlap = 0.1  # Small overlap to prevent line
            total_width = new_width1 + new_width2 - overlap
            new_page = output.new_page(width=total_width, height=target_height)
            
            # Create a white background to ensure clean joining
            new_page.draw_rect(new_page.rect, color=(1, 1, 1), fill=(1, 1, 1))
            
            # Create temporary documents at the right scale
            temp_doc1 = fitz.open()
            temp_page1 = temp_doc1.new_page(width=new_width1, height=target_height)
            temp_page1.show_pdf_page(temp_page1.rect, doc1, i)
            
            temp_doc2 = fitz.open()
            temp_page2 = temp_doc2.new_page(width=new_width2, height=target_height)
            temp_page2.show_pdf_page(temp_page2.rect, doc2, i)
            
            # Insert the pages side by side with precise positioning to eliminate the line
            # Adjust the positioning to ensure no gap between the PDFs
            new_page.show_pdf_page(fitz.Rect(0, 0, new_width2, target_height), temp_doc2, 0)  # Left side
            new_page.show_pdf_page(fitz.Rect(new_width2-overlap, 0, total_width, target_height), temp_doc1, 0)  # Right side with slight overlap
            
            # Close temporary documents
            temp_doc1.close()
            temp_doc2.close()
        
        # Save the combined PDF
        output.save(output_path)
        print(f"Combined PDF saved to {output_path}")
        
        # Close all documents
        doc1.close()
        doc2.close()
        output.close()
        
        return output_path
        
    except Exception as e:
        print(f"Error combining PDFs: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_args():
    parser = argparse.ArgumentParser(description='Combine two PDFs side by side')
    parser.add_argument('--pdf1_path', type=str, default='outputs_main_results/model_performance_dotted_comparison_ty.pdf', help='Path to the first PDF file')
    parser.add_argument('--pdf2_path', type=str, default='/Users/tianyu/Work/paper2code/model_paper_knowledge_debug_ty.pdf', help='Path to the second PDF file')
    parser.add_argument('--output_path', type=str, default='outputs_main_results/combined_visualization.pdf', help='Path to save the combined PDF')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    output_dir = 'outputs_main_results'
    os.makedirs(output_dir, exist_ok=True)
    combine_pdfs_side_by_side(args.pdf1_path, args.pdf2_path, args.output_path)


