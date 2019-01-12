program matmulprogram
implicit none

      real, dimension(10, 10) :: w1
      real, dimension(10, 1) :: w2, h1, input
      real, dimension(1, 1) :: h2

      integer :: i, j
      real :: startT, endT, execTime, h2_error
      
      do i = 1, 10 
        do j = 1, 10
                w1(i, j) = 1
        end do
        w2(i, 1) = 1
        input(i, 1) = 1
      end do

c      call cpu_time(startT)
      
c      do i= 1, 1000
              h1 = matmul(w1, input)
              where (h1<0) h1 = 0
              h2 = matmul(transpose(h1), w2)
c      end do    
      
c      call cpu_time(endT)

c      print*, endT - startT
       
       h2_error = 101 - h2

       h2_delta = h2_error*h2

       h1_error = h2_delta*w2

       h1_delta = h1_error * where (h1>0)
       
       w1 += 0.001*h1.dot(h2_delta)
       w2 += 0.001*input.dot(h1_delta)
       

end program matmulprogram

