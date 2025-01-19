import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import '../css/Home.css';
import { API_URL, model_URL } from '../data/apipath';

function Home() {
    const Navigate = useNavigate();
    const [navOpen, setNavOpen] = useState(false);
    const [currentIndex, setCurrentIndex] = useState(0);
    const leftImages = ['img/left3.jpg', 'img/left1.jpg', 'img/left4.webp'];
    const rightImages = ['img/right3.avif', 'img/right1.jpeg', 'img/necklace.jpg'];
  
    useEffect(() => {
      const activatemodel = async () => {
        try {
          const status = await fetch(`${model_URL}`, { method: 'GET' });
          if (!status.ok) console.log("Model is down");
        } catch (err) {
          console.log(err);
        }
      };
  
      const activatebacked = async () => {
        try {
          const status = await fetch(`${API_URL}/`, { method: 'HEAD' });
          if (!status.ok) console.log("Backend is down");
        } catch (err) {
          console.log(err);
        }
      };
  
      activatemodel();
      activatebacked();
  
      const interval = setInterval(() => {
        setCurrentIndex((prevIndex) => (prevIndex + 1) % leftImages.length);
      }, 3000);
  
      const observerOptions = { root: null, threshold: 0.1, rootMargin: "0px" };
  
      const observeElements = (selector, className) => {
        const observer = new IntersectionObserver(entries => {
          entries.forEach(entry => {
            if (entry.isIntersecting) {
              entry.target.classList.add(className);
            } else {
              entry.target.classList.remove(className);
            }
          });
        }, observerOptions);
  
        document.querySelectorAll(selector).forEach(el => observer.observe(el));
      };
  
      observeElements('.hidden', 'show');
      observeElements('.zoom-out', 'active-zoom');
      observeElements('.translate', 'translated');
      observeElements('.ltranslate', 'lefttranslated');
      observeElements('.explorebtn', 'animate__heartBeat');
  
      return () => clearInterval(interval);
    }, [leftImages.length]);
  

  return (
    <>
      <header>
      <nav id="nav" className="nav">
      <img src={'ed.jpg'} alt="logo" className="imglogo" />
        <div className="logo" id="logo">Elite Designs</div>
        <button
          className="hamburger"
          onClick={() =>{ setNavOpen(!navOpen);console.log("Hamburger clicked. navOpen state:", !navOpen);}}
          aria-expanded={navOpen}
          aria-controls="navitems"
        >
          ☰
        </button>
        <div id="navitems" className={!navOpen ? 'navitems' : 'notnavitems'} >
          <ul>
            <li><Link to="/" className="active">Home</Link></li>
            <li><Link to="/explore">Explore</Link></li>
            <li><Link to="/contact">Contact Us</Link></li>
            <button onClick={() => Navigate('/login')} className="uploadbtn">Get Started</button>
          </ul>
        </div>
      </nav>

      </header>
      <div className="intro-slide">
        <p className="headingname">ELITE</p>
        <p className="introname">The Jewelry Designer</p>
        <video id="Video" autoPlay muted loop>
          <source src="img/vido.mp4" type="video/mp4" />
        </video>
      </div>
      
      <div className="container0 slide next-slide">
        <div className="overlay" />
        <div id="matter0" className="zoom-out" >
          <p id="l01">Unlock the true potential of your jewelry with our advanced image enhancement technology</p>
          <h1 id="l02">Transform Your Jewelry Photos into Stunning Creations</h1>
          <p id="l03">Upload a basic image of your jewelry, and let our AI-powered platform work its magic</p>
          <button id="btn0"onClick={() => Navigate('/login')}>Get Started</button>
        </div>
      </div>

      <div className="container1" style={{ backgroundColor: "black" }}>
        <div className="c1 ltranslate" id="matter1">
          <h3>Visualize Jewelry Designs Instantly</h3>
          <p>Upload a basic image and generate high-quality jewelry images</p>
          <button className="c1" onClick={() => window.location.href='/login'}>Generate</button>
        </div>
        <div className="c1 translate" id="image11">
          <img src="img/im1.jpg" alt="none" />
        </div>
      </div>

      <div className="container2">
        <div className="left2 c2item1 c2items c2img" id="left2" style={{ backgroundImage: `url(${leftImages[currentIndex]})` }} >
            {/* <button class="c2btn1 button " id="c2btn1"></button>
            <button class="c2btn2 button" id="c2btn2"></button>
            <button class="c2btn3 button" id="c2btn3"></button> */}
        </div>
        <div className="middle2 c2item2 c2items" id="middle">
          <p id="c2l1">Elite</p>
          <p id="c2l2">presents</p>
          <p id="c2l3">Jewelry Collections</p>
          <p id="c2l4">Explore the Pinnacle of AI-Created Jewelry</p>
          <p id="c2l5">Dive into a curated collection where technology meets artistry, showcasing stunning designs crafted from your visions.</p>
          <button className="explorebtn" onClick={() => window.location.href='/login'}>Explore Now</button>
        </div>
        <div className="right2 c2items c2item3 c2img" id="right2" style={{ backgroundImage: `url(${rightImages[currentIndex]})` }} >
            {/* <button class="c2btn4 button" id="c2btn4"></button>
            <button class="c2btn5 button" id="c2btn5"></button>
            <button class="c2btn6 button" id="c2btn6"></button> */}
        </div>
      </div>

      <div className="contact c3 container3" id="contact" style={{ backgroundColor: "black" ,color:"white"}}>
        <h4>ELITE</h4>
        <p>We are a team of innovative college students turning basic jewelry concepts into extraordinary designs.</p>
        <button onClick={() => window.location.href='/login'}>GET IN TOUCH →</button>
      </div>
      <footer style={{ backgroundColor: "black" }}>
      <button className="qr-btn">
          <img src="qr.jpg" alt="QR Code" style={{ width: '100px' }} />
        </button>
        <div className="footer">
          <p className='copyright'>©2024 Elite Designs</p>
          <p className="socialmedia">E-mail, Instagram, X</p>
          <p className='mail'>elitedesigns.g169@gmail.com</p>
        </div>
      </footer>
      

    </>
  );
}
export default Home;